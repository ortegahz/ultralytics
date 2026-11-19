// ===========================================================================
// mosaic_track_fusion.cpp -- 时空融合后处理的 C++ 移植（实现）
//
//   参考：ultralytics/manu/evaluation/eval_bidirectional_track_fusion.py
//         ultralytics/manu/evaluation/system_sota_longquan.py
//   冻结参数：system_sota_longquan_trial0080_delivery_params.json（th_base=0.45）
//
//   本文件**零第三方依赖**：只用 <cmath> / <algorithm> / <vector> / <map>。
//   ⇒ host 上可单独编译做精度对齐；板卡上可原生编译。
//
//   数值保真说明（详见 mosaic_track_fusion.h 顶部）：
//     * numpy float32 数组的运算一律按 float32 走；
//     * numpy 标量与 Python float 比较时按 float32 比（NEP 50）⇒ 阈值先降精度；
//     * `np.linalg.norm` = sqrt(add.reduce(x*x))：float32 乘 + float32 加 + float32 sqrt；
//     * `np.var` / `np.mean` 用 numpy 的 pairwise 求和口径复刻；
//     * 卡尔曼矩阵乘用 double 累加、float32 存回（贴近 BLAS sgemm 的 FMA 结果）。
// ===========================================================================
#include "mosaic_track_fusion.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace mosaic {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// ---------------------------------------------------------------------------
// numpy 语义的原语
// ---------------------------------------------------------------------------

// numpy: np.add.reduce on a float32 array -> pairwise summation.
// 复刻 numpy 的 pairwise_sum_FLOAT（n<8 顺序累加；n<=128 八路展开；再大则二分）。
double NpPairwiseSumF(const float* a, size_t n) {
  if (n < 8) {
    float res = 0.0f;
    for (size_t i = 0; i < n; ++i) res += a[i];
    return (double)res;
  }
  if (n <= 128) {
    float r[8];
    for (int i = 0; i < 8; ++i) r[i] = a[i];
    size_t i = 8;
    for (; i + 8 <= n; i += 8) {
      r[0] += a[i + 0];
      r[1] += a[i + 1];
      r[2] += a[i + 2];
      r[3] += a[i + 3];
      r[4] += a[i + 4];
      r[5] += a[i + 5];
      r[6] += a[i + 6];
      r[7] += a[i + 7];
    }
    float res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += a[i];
    return (double)res;
  }
  size_t n2 = n / 2;
  n2 -= n2 % 8;
  return NpPairwiseSumF(a, n2) + NpPairwiseSumF(a + n2, n - n2);
}

// numpy: x.mean() on a float32 array  ->  float32
float NpMeanF(const std::vector<float>& v) {
  if (v.empty()) return 0.0f;
  const double s = NpPairwiseSumF(v.data(), v.size());
  return (float)(s / (double)v.size());
}

// numpy: np.var(a) on a float32 array (ddof=0)  ->  float32
float NpVarF(const std::vector<float>& v) {
  if (v.empty()) return 0.0f;
  const float mean = NpMeanF(v);
  std::vector<float> sq(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    const float d = v[i] - mean;
    sq[i] = d * d;
  }
  const double s = NpPairwiseSumF(sq.data(), sq.size());
  return (float)(s / (double)v.size());
}

// numpy: np.linalg.norm(v) for a 1-D float32 array  ->  float32
//   实现是 x.dot(x)（BLAS sdot，float32 累加）再开方；这里用 double 累加后
//   舍入到 float32，等价于 FMA 的「一次舍入」结果。
float NpNorm1Df(float ax, float ay) {
  const double s = (double)ax * (double)ax + (double)ay * (double)ay;
  return std::sqrt((float)s);
}

// numpy: np.linalg.norm(x, axis=-1) for a float32 array -> float32
//   实现是 sqrt(add.reduce(x*x))：乘法 float32、求和 float32、开方 float32。
float NpNorm2Df(float ax, float ay) {
  const float sx = ax * ax;
  const float sy = ay * ay;
  const float s = sx + sy;
  return std::sqrt(s);
}

// ---------------------------------------------------------------------------
// 矩阵小工具（4x4 / 2x2）：内部 double 累加，float32 存回。
// ---------------------------------------------------------------------------
template <int N, int M, int K>
void MatMul(const float a[N][K], const float b[K][M], float out[N][M]) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      double s = 0.0;
      for (int k = 0; k < K; ++k) s += (double)a[i][k] * (double)b[k][j];
      out[i][j] = (float)s;
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// 矩形线性指派：scipy.optimize.linear_sum_assignment 的等价实现。
//
//   移植自 scipy 1.15.3 的 scipy/optimize/rectangular_lsap/rectangular_lsap.cpp
//   （Crouse, "On implementing 2D rectangular assignment algorithms",
//    IEEE TAES 52(4):1679-1796, 2016）。scipy 的 Python 包装层用
//   PyArray_ContiguousFromAny(..., NPY_DOUBLE, ...) ⇒ **代价矩阵一律先上转
//   float64** 再求解，这里保持一致。
// ---------------------------------------------------------------------------
void LinearSumAssignment(const double* cost_in, int nrow, int ncol, std::vector<int32_t>* row_ind,
                         std::vector<int32_t>* col_ind) {
  row_ind->clear();
  col_ind->clear();
  if (nrow <= 0 || ncol <= 0) return;

  int nr = nrow;
  int nc = ncol;
  const bool transpose = nc < nr;
  std::vector<double> temp;
  const double* cost = cost_in;
  if (transpose) {
    temp.resize((size_t)nr * (size_t)nc);
    for (int i = 0; i < nr; ++i) {
      for (int j = 0; j < nc; ++j) temp[(size_t)j * (size_t)nr + (size_t)i] = cost_in[(size_t)i * (size_t)nc + (size_t)j];
    }
    std::swap(nr, nc);
    cost = temp.data();
  }

  std::vector<double> u((size_t)nr, 0.0);
  std::vector<double> v((size_t)nc, 0.0);
  std::vector<double> shortest((size_t)nc, 0.0);
  std::vector<int> path((size_t)nc, -1);
  std::vector<int> col4row((size_t)nr, -1);
  std::vector<int> row4col((size_t)nc, -1);
  std::vector<char> sr((size_t)nr, 0);
  std::vector<char> sc((size_t)nc, 0);
  std::vector<int> remaining((size_t)nc, 0);

  for (int cur_row = 0; cur_row < nr; ++cur_row) {
    // ---- augmenting_path ----
    double min_val = 0.0;
    int num_remaining = nc;
    // 逆序初始化 ⇒ 常数矩阵的解是单位阵（与 scipy 一致）
    for (int it = 0; it < nc; ++it) remaining[(size_t)it] = nc - it - 1;
    std::fill(sr.begin(), sr.end(), 0);
    std::fill(sc.begin(), sc.end(), 0);
    std::fill(shortest.begin(), shortest.end(), kInf);

    int i = cur_row;
    int sink = -1;
    while (sink == -1) {
      int index = -1;
      double lowest = kInf;
      sr[(size_t)i] = 1;
      for (int it = 0; it < num_remaining; ++it) {
        const int j = remaining[(size_t)it];
        const double r = min_val + cost[(size_t)i * (size_t)nc + (size_t)j] - u[(size_t)i] - v[(size_t)j];
        if (r < shortest[(size_t)j]) {
          path[(size_t)j] = i;
          shortest[(size_t)j] = r;
        }
        if (shortest[(size_t)j] < lowest ||
            (shortest[(size_t)j] == lowest && row4col[(size_t)j] == -1)) {
          lowest = shortest[(size_t)j];
          index = it;
        }
      }
      min_val = lowest;
      if (min_val == kInf) break;  // 不可行（理论上不会发生：距离矩阵有限）
      const int j = remaining[(size_t)index];
      if (row4col[(size_t)j] == -1) {
        sink = j;
      } else {
        i = row4col[(size_t)j];
      }
      sc[(size_t)j] = 1;
      remaining[(size_t)index] = remaining[(size_t)(--num_remaining)];
    }
    if (sink < 0) break;

    // ---- 更新对偶变量 ----
    u[(size_t)cur_row] += min_val;
    for (int r = 0; r < nr; ++r) {
      if (sr[(size_t)r] && r != cur_row) {
        u[(size_t)r] += min_val - shortest[(size_t)col4row[(size_t)r]];
      }
    }
    for (int j = 0; j < nc; ++j) {
      if (sc[(size_t)j]) v[(size_t)j] -= min_val - shortest[(size_t)j];
    }

    // ---- 增广 ----
    int j = sink;
    for (;;) {
      const int r = path[(size_t)j];
      row4col[(size_t)j] = r;
      std::swap(col4row[(size_t)r], j);
      if (r == cur_row) break;
    }
  }

  if (transpose) {
    std::vector<int> idx((size_t)nr);
    for (int k = 0; k < nr; ++k) idx[(size_t)k] = k;
    std::sort(idx.begin(), idx.end(), [&col4row](int a, int b) { return col4row[(size_t)a] < col4row[(size_t)b]; });
    for (int k = 0; k < nr; ++k) {
      row_ind->push_back((int32_t)col4row[(size_t)idx[(size_t)k]]);
      col_ind->push_back((int32_t)idx[(size_t)k]);
    }
  } else {
    for (int r = 0; r < nr; ++r) {
      row_ind->push_back((int32_t)r);
      col_ind->push_back((int32_t)col4row[(size_t)r]);
    }
  }
}

// ---------------------------------------------------------------------------
// TrackFusionEngine
// ---------------------------------------------------------------------------

TrackFusionEngine::TrackFusionEngine(const TrackFusionParams& params) : params_(params) { Reset(); }

void TrackFusionEngine::Reset() {
  next_track_id_ = 1;
  num_frames_ = 0;
  last_step_frame_ = -1;
  active_.clear();
  finished_.clear();
  stitched_.clear();
  fused_.clear();
}

void TrackFusionEngine::InitTrackMatrices(Track* t) const {
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      t->P[i][j] = 0.0f;
      t->Q[i][j] = 0.0f;
      t->F[i][j] = 0.0f;
    }
  }
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) t->H[i][j] = 0.0f;
    for (int j = 0; j < 2; ++j) t->R[i][j] = 0.0f;
  }
  t->P[0][0] = 10.0f; t->P[1][1] = 10.0f; t->P[2][2] = 50.0f; t->P[3][3] = 50.0f;
  t->Q[0][0] = 1.0f;  t->Q[1][1] = 1.0f;  t->Q[2][2] = 4.0f;  t->Q[3][3] = 4.0f;
  t->R[0][0] = 4.0f;  t->R[1][1] = 4.0f;
  t->F[0][0] = 1.0f; t->F[0][2] = 1.0f;
  t->F[1][1] = 1.0f; t->F[1][3] = 1.0f;
  t->F[2][2] = 1.0f;
  t->F[3][3] = 1.0f;
  t->H[0][0] = 1.0f;
  t->H[1][1] = 1.0f;
}

void TrackFusionEngine::Predict(Track* t, int64_t frame_idx, int64_t dt) const {
  // ---- Δn（物理帧间隔）感知的状态转移 ----
  //   dt == 1：F_dt ≡ t->F、Q_scaled ≡ t->Q ⇒ 与 legacy 逐位一致；
  //   dt >  1：位置项按 dt 缩放（F_dt[0][2] = F_dt[1][3] = dt），
  //             过程噪声按 Q_scaled = Q * dt 放大。
  float Fdt[4][4];
  std::memcpy(Fdt, t->F, sizeof(Fdt));
  if (dt > 1) {
    Fdt[0][2] = (float)dt;
    Fdt[1][3] = (float)dt;
  }
  float Qs[4][4];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) Qs[i][j] = (float)((double)t->Q[i][j] * (double)dt);
  }

  // x = F_dt @ x
  float nx[4];
  for (int i = 0; i < 4; ++i) {
    double s = 0.0;
    for (int k = 0; k < 4; ++k) s += (double)Fdt[i][k] * (double)t->x[k];
    nx[i] = (float)s;
  }
  std::memcpy(t->x, nx, sizeof(nx));

  // P = F_dt @ P @ F_dt.T + Q_scaled
  float fp[4][4];
  MatMul<4, 4, 4>(Fdt, t->P, fp);
  float fpt[4][4];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      double s = 0.0;
      for (int k = 0; k < 4; ++k) s += (double)fp[i][k] * (double)Fdt[j][k];  // (F.T)[k][j] = F[j][k]
      fpt[i][j] = (float)s;
    }
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) t->P[i][j] = (float)((double)fpt[i][j] + (double)Qs[i][j]);
  }

  t->age += 1;
  t->time_since_update += 1;
  t->last_frame = frame_idx;
}

void TrackFusionEngine::Update(Track* t, const float pos[2], double score, int64_t frame_idx) const {
  t->time_since_update = 0;
  t->hits += 1;
  // ---- 分数混合：Δn 感知 ----
  //   alpha_decay == 1.0 ⇒ legacy 公式（逐位一致）；
  //   否则按「距上次命中的物理帧数」做指数衰减。
  if (params_.alpha_decay == 1.0) {
    t->score = 0.65 * t->score + 0.35 * score;
  } else {
    int64_t dt_hit = frame_idx - t->last_hit_fid;
    if (dt_hit < 1) dt_hit = 1;
    t->score = 0.65 * (t->score * std::pow(params_.alpha_decay, (double)dt_hit)) + 0.35 * score;
  }
  t->last_hit_fid = frame_idx;
  t->last_frame = frame_idx;
  t->last_observed_frame = frame_idx;

  // H @ x   (2,)
  float hx[2];
  {
    double s0 = 0.0, s1 = 0.0;
    for (int k = 0; k < 4; ++k) {
      s0 += (double)t->H[0][k] * (double)t->x[k];
      s1 += (double)t->H[1][k] * (double)t->x[k];
    }
    hx[0] = (float)s0;
    hx[1] = (float)s1;
  }
  // y = pos - H @ x    (float32)
  const float yv[2] = {pos[0] - hx[0], pos[1] - hx[1]};

  // S = H @ P @ H.T + R
  float hp[2][4];
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 4; ++j) {
      double s = 0.0;
      for (int k = 0; k < 4; ++k) s += (double)t->H[i][k] * (double)t->P[k][j];
      hp[i][j] = (float)s;
    }
  }
  float smat[2][2];
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      double s = 0.0;
      for (int k = 0; k < 4; ++k) s += (double)hp[i][k] * (double)t->H[j][k];  // (H.T)[k][j] = H[j][k]
      smat[i][j] = (float)((double)s + (double)t->R[i][j]);
    }
  }

  // inv(S)：numpy 走 LAPACK sgetrf/sgetri；这里用解析式（double 累加后舍入）
  const double det = (double)smat[0][0] * (double)smat[1][1] - (double)smat[0][1] * (double)smat[1][0];
  float sinv[2][2];
  sinv[0][0] = (float)((double)smat[1][1] / det);
  sinv[0][1] = (float)(-(double)smat[0][1] / det);
  sinv[1][0] = (float)(-(double)smat[1][0] / det);
  sinv[1][1] = (float)((double)smat[0][0] / det);

  // K = P @ H.T @ inv(S)
  float pht[4][2];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      double s = 0.0;
      for (int k = 0; k < 4; ++k) s += (double)t->P[i][k] * (double)t->H[j][k];  // (H.T)[k][j] = H[j][k]
      pht[i][j] = (float)s;
    }
  }
  float kmat[4][2];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 2; ++j) {
      double s = 0.0;
      for (int k = 0; k < 2; ++k) s += (double)pht[i][k] * (double)sinv[k][j];
      kmat[i][j] = (float)s;
    }
  }

  // x = x + K @ y
  for (int i = 0; i < 4; ++i) {
    double s = 0.0;
    for (int j = 0; j < 2; ++j) s += (double)kmat[i][j] * (double)yv[j];
    t->x[i] = (float)((double)t->x[i] + s);
  }

  // P = (I - K @ H) @ P
  float kh[4][4];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      double s = 0.0;
      for (int k = 0; k < 2; ++k) s += (double)kmat[i][k] * (double)t->H[k][j];  // (K@H)[i][j]
      kh[i][j] = (float)s;
    }
  }
  float amat[4][4];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      const float ident = (i == j) ? 1.0f : 0.0f;
      amat[i][j] = ident - kh[i][j];
    }
  }
  float newp[4][4];
  MatMul<4, 4, 4>(amat, t->P, newp);
  std::memcpy(t->P, newp, sizeof(newp));

  Observation obs;
  obs.pos[0] = pos[0];
  obs.pos[1] = pos[1];
  obs.score = score;
  obs.measured = true;
  t->observations[frame_idx] = obs;
}

float TrackFusionEngine::VelocityNorm(const Track& t) const {
  return NpNorm1Df(t.x[2], t.x[3]);
}

double TrackFusionEngine::NetDisplacement(const Track& t) const {
  return (double)NpNorm1Df(t.x[0] - t.start_pos[0], t.x[1] - t.start_pos[1]);
}

bool TrackFusionEngine::IsConfirmed(const Track& t) const {
  return t.is_confirmed || (t.hits >= params_.min_hits && t.score >= params_.min_track_score);
}

void TrackFusionEngine::Step(int64_t frame_idx, const TrackFusionCand* cands, size_t count) {
  // ---- 0. 乱序保护 + 物理帧间隔 dt ----
  //     frame_idx 必须严格递增：frame_idx <= 上一次被接受的帧号 ⇒ 整次忽略
  //     （状态完全不变）。dt = 本次与上一次的物理帧号之差（首次为 1）。
  if (frame_idx <= last_step_frame_) return;
  const int64_t dt = (last_step_frame_ < 0) ? 1 : (frame_idx - last_step_frame_);
  last_step_frame_ = frame_idx;

  if (frame_idx + 1 > num_frames_) num_frames_ = frame_idx + 1;

  // ---- 1. 天地分区 + 双层阈值 ----
  //     numpy: is_sky = pred_pts[:,1] < img_h * sky_ratio  （float32 比较）
  const float sky_boundary = (float)((double)params_.img_h * params_.sky_ratio);
  const float th_base = (float)params_.th_base;
  const float th_salvage = (float)params_.th_salvage;
  const float th_ground = (float)params_.th_ground;
  const float th_deep = (float)params_.th_deep;

  std::vector<float> high_x, high_y, high_s;
  std::vector<float> salv_x, salv_y, salv_s;
  std::vector<float> deep_x, deep_y, deep_s;

  for (size_t i = 0; i < count; ++i) {
    const float x = cands[i].x;
    const float y = cands[i].y;
    const float s = cands[i].score;
    const bool is_sky = y < sky_boundary;
    const bool high = is_sky ? (s >= th_base) : (s >= th_ground);
    const bool salvage = is_sky && (s >= th_salvage) && (s < th_base);
    if (high) {
      high_x.push_back(x);
      high_y.push_back(y);
      high_s.push_back(s);
    }
    if (salvage) {
      salv_x.push_back(x);
      salv_y.push_back(y);
      salv_s.push_back(s);
    }
    if (th_deep > 0.0f && is_sky && s >= th_deep && s < std::min(th_salvage, th_base)) {
      deep_x.push_back(x);
      deep_y.push_back(y);
      deep_s.push_back(s);
    }
  }

  // ---- 2. 密集杂波簇过滤（只作用于 salvage 池）----
  //    numpy: filter_dense_clutter_clusters(cluster_radius=25.0, max_neighbors=2)
  {
    const size_t n = salv_x.size();
    if (n > 2) {  // `if len(pts) <= max_neighbors: return`
      std::vector<char> keep(n, 0);
      for (size_t i = 0; i < n; ++i) {
        int neighbors = 0;
        for (size_t j = 0; j < n; ++j) {
          const float ddx = salv_x[i] - salv_x[j];
          const float ddy = salv_y[i] - salv_y[j];
          // numpy: sqrt(sum(diff**2)) in float32, 与 25.0 比较时降成 float32
          const float dd = NpNorm2Df(ddx, ddy);
          if (dd < 25.0f) ++neighbors;
        }
        const int neighbor_counts = neighbors - 1;
        keep[i] = (neighbor_counts <= 2) ? 1 : 0;
      }
      std::vector<float> kx, ky, ks;
      for (size_t i = 0; i < n; ++i) {
        if (!keep[i]) continue;
        kx.push_back(salv_x[i]);
        ky.push_back(salv_y[i]);
        ks.push_back(salv_s[i]);
      }
      salv_x.swap(kx);
      salv_y.swap(ky);
      salv_s.swap(ks);
    }
  }

  // ---- 3. 卡尔曼预测 ----
  for (Track& t : active_) Predict(&t, frame_idx, dt);

  std::vector<char> matched_tracks(active_.size(), 0);
  std::vector<char> matched_high(high_x.size(), 0);

  // ---- Step A: 已有轨迹 ↔ 高置信检测 ----
  if (!active_.empty() && !high_x.empty()) {
    const size_t nr = active_.size();
    const size_t nc = high_x.size();
    std::vector<double> dists(nr * nc);
    for (size_t r = 0; r < nr; ++r) {
      const float tx = active_[r].x[0];
      const float ty = active_[r].x[1];
      for (size_t c = 0; c < nc; ++c) {
        const float dx = tx - high_x[c];
        const float dy = ty - high_y[c];
        dists[r * nc + c] = (double)NpNorm2Df(dx, dy);
      }
    }
    std::vector<int32_t> ri, ci;
    LinearSumAssignment(dists.data(), (int)nr, (int)nc, &ri, &ci);
    for (size_t k = 0; k < ri.size(); ++k) {
      const size_t r = (size_t)ri[k];
      const size_t c = (size_t)ci[k];
      Track& trk = active_[r];
      const double gate = std::min(params_.max_match_dist,
                                   params_.match_dist + params_.r_slope * (double)(dt - 1) +
                                       0.5 * (double)VelocityNorm(trk));
      // numpy: float32 距离 与 Python float 门限比较 ⇒ 门限降到 float32
      if (dists[r * nc + c] <= (double)(float)gate) {
        const float pos[2] = {high_x[c], high_y[c]};
        Update(&trk, pos, (double)high_s[c], frame_idx);
        matched_tracks[r] = 1;
        matched_high[c] = 1;
      }
    }
  }

  // ---- Step B: 未匹配轨迹 ↔ salvage 池 ----
  std::vector<size_t> unmatched;
  for (size_t i = 0; i < active_.size(); ++i) {
    if (!matched_tracks[i]) unmatched.push_back(i);
  }
  if (!unmatched.empty() && !salv_x.empty()) {
    const size_t nr = unmatched.size();
    const size_t nc = salv_x.size();
    std::vector<double> dists(nr * nc);
    for (size_t r = 0; r < nr; ++r) {
      const Track& trk = active_[unmatched[r]];
      const float tx = trk.x[0];
      const float ty = trk.x[1];
      for (size_t c = 0; c < nc; ++c) {
        const float dx = tx - salv_x[c];
        const float dy = ty - salv_y[c];
        dists[r * nc + c] = (double)NpNorm2Df(dx, dy);
      }
    }
    std::vector<int32_t> ri, ci;
    LinearSumAssignment(dists.data(), (int)nr, (int)nc, &ri, &ci);
    for (size_t k = 0; k < ri.size(); ++k) {
      const size_t r = (size_t)ri[k];
      const size_t c = (size_t)ci[k];
      const size_t orig = unmatched[r];
      Track& trk = active_[orig];
      const double gate = std::min(params_.max_match_dist,
                                   params_.match_dist + params_.r_slope * (double)(dt - 1) +
                                       0.4 * (double)VelocityNorm(trk));
      if (dists[r * nc + c] <= (double)(float)gate) {
        const float pos[2] = {salv_x[c], salv_y[c]};
        Update(&trk, pos, (double)salv_s[c], frame_idx);
        matched_tracks[orig] = 1;
      }
    }
  }

  // ---- Step B2: 深潜打捞（th_deep > 0 才启用）----
  if (th_deep > 0.0f && !deep_x.empty()) {
    std::vector<size_t> mature;
    for (size_t i = 0; i < active_.size(); ++i) {
      if (!matched_tracks[i] && active_[i].hits >= params_.min_hits_deep) mature.push_back(i);
    }
    if (!mature.empty()) {
      const size_t nr = mature.size();
      const size_t nc = deep_x.size();
      std::vector<double> dists(nr * nc);
      for (size_t r = 0; r < nr; ++r) {
        const Track& trk = active_[mature[r]];
        const float tx = trk.x[0];
        const float ty = trk.x[1];
        for (size_t c = 0; c < nc; ++c) {
          const float dx = tx - deep_x[c];
          const float dy = ty - deep_y[c];
          dists[r * nc + c] = (double)NpNorm2Df(dx, dy);
        }
      }
      std::vector<int32_t> ri, ci;
      LinearSumAssignment(dists.data(), (int)nr, (int)nc, &ri, &ci);
      for (size_t k = 0; k < ri.size(); ++k) {
        const size_t r = (size_t)ri[k];
        const size_t c = (size_t)ci[k];
        const size_t orig = mature[r];
        Track& trk = active_[orig];
        if (dists[r * nc + c] <= (double)(float)params_.match_dist) {
          const float pos[2] = {deep_x[c], deep_y[c]};
          Update(&trk, pos, (double)deep_s[c], frame_idx);
          matched_tracks[orig] = 1;
        }
      }
    }
  }

  // ---- Step C: 只有未匹配的**高置信**检测才能开新轨 ----
  for (size_t i = 0; i < high_x.size(); ++i) {
    if (matched_high[i]) continue;
    Track t;
    t.id = next_track_id_++;
    t.x[0] = high_x[i];
    t.x[1] = high_y[i];
    t.x[2] = 0.0f;
    t.x[3] = 0.0f;
    InitTrackMatrices(&t);
    t.start_frame = frame_idx;
    t.last_frame = frame_idx;
    t.last_observed_frame = frame_idx;
    t.last_hit_fid = frame_idx;
    t.hits = 1;
    t.age = 1;
    t.time_since_update = 0;
    t.score = (double)high_s[i];
    t.init_score = t.score;
    t.is_confirmed = (high_s[i] >= (float)params_.instant_conf);
    t.start_pos[0] = high_x[i];
    t.start_pos[1] = high_y[i];
    Observation obs;
    obs.pos[0] = high_x[i];
    obs.pos[1] = high_y[i];
    obs.score = t.score;
    obs.measured = true;
    t.observations[frame_idx] = obs;
    active_.push_back(std::move(t));
  }

  // ---- Step D: 淘汰死轨 + 地面静态杂波剔除 ----
  std::vector<Track> surviving;
  for (Track& t : active_) {
    const bool dead = (params_.max_age_phys >= 0)
                          ? ((frame_idx - t.last_hit_fid) > params_.max_age_phys)
                          : (t.time_since_update > params_.max_age);
    if (dead) {
      const float sky_boundary = (float)((double)params_.img_h * params_.sky_ratio);
      const bool is_in_sky = t.x[1] < sky_boundary;
      if (!is_in_sky && t.hits >= 5 && params_.min_displacement > 0.0) {
        if (NetDisplacement(t) < params_.min_displacement) continue;  // 地面静态杂波
      }
      finished_.push_back(std::move(t));
    } else {
      surviving.push_back(std::move(t));
    }
  }
  active_.swap(surviving);
}

void TrackFusionEngine::OnlineFused(int64_t frame_idx, std::vector<TrackFusionDet>* out) const {
  out->clear();
  for (const Track& t : active_) {
    auto it = t.observations.find(frame_idx);
    if (it == t.observations.end()) continue;
    if (!IsConfirmed(t)) continue;
    TrackFusionDet d;
    d.track_id = t.id;
    d.x = it->second.pos[0];
    d.y = it->second.pos[1];
    d.score = it->second.score;
    d.infilled = false;
    out->push_back(d);
  }
  std::stable_sort(out->begin(), out->end(),
                   [](const TrackFusionDet& a, const TrackFusionDet& b) { return a.score > b.score; });
}

void TrackFusionEngine::Finalize(std::vector<std::vector<TrackFusionDet>>* out) {
  out->clear();
  const int64_t num_frames = num_frames_;

  // ---- finalize(): 收尾未死透的轨迹（同样做地面静态剔除）----
  {
    const float sky_boundary = (float)((double)params_.img_h * params_.sky_ratio);
    for (Track& t : active_) {
      const bool is_in_sky = t.x[1] < sky_boundary;
      if (!is_in_sky && t.hits >= 5 && params_.min_displacement > 0.0) {
        if (NetDisplacement(t) < params_.min_displacement) continue;
      }
      finished_.push_back(std::move(t));
    }
    active_.clear();
  }

  // ---- Pass 2a: 碎片缝合 ----
  std::vector<Track> all = std::move(finished_);
  finished_.clear();
  std::stable_sort(all.begin(), all.end(),
                   [](const Track& a, const Track& b) { return a.start_frame < b.start_frame; });
  std::vector<Track> merged;
  for (Track& trk : all) {
    bool matched = false;
    for (Track& prev : merged) {
      const int64_t gap = trk.start_frame - prev.last_observed_frame;
      if (gap < 1 || gap > params_.stitch_gap) continue;
      // 外推位置 = prev 末次观测位置 + prev 速度 * gap
      auto it = prev.observations.find(prev.last_observed_frame);
      if (it == prev.observations.end()) continue;
      const float px = it->second.pos[0];
      const float py = it->second.pos[1];
      const float vx = prev.x[2];
      const float vy = prev.x[3];
      const float ex = px + vx * (float)gap;
      const float ey = py + vy * (float)gap;
      auto it2 = trk.observations.find(trk.start_frame);
      if (it2 == trk.observations.end()) continue;
      const float spatial = NpNorm2Df(ex - it2->second.pos[0], ey - it2->second.pos[1]);
      const double allowed = std::max(params_.stitch_max_dist, 10.0 * (double)gap);
      if ((double)spatial <= (double)(float)allowed) {
        for (const auto& kv : trk.observations) prev.observations[kv.first] = kv.second;
        prev.hits += trk.hits;
        prev.last_frame = std::max(prev.last_frame, trk.last_frame);
        prev.last_observed_frame = std::max(prev.last_observed_frame, trk.last_observed_frame);
        prev.score = std::max(prev.score, trk.score);
        prev.is_confirmed = prev.is_confirmed || trk.is_confirmed;
        matched = true;
        break;
      }
    }
    if (!matched) merged.push_back(std::move(trk));
  }
  stitched_ = merged;

  // ---- Pass 2b: 双向平滑 / 插补 / 刚性坏点剪枝 ----
  std::vector<std::vector<TrackFusionDet>> frame_outputs((size_t)std::max<int64_t>(num_frames, 0));
  for (const Track& trk : stitched_) {
    if (!IsConfirmed(trk)) continue;
    std::vector<int64_t> obs_frames;
    obs_frames.reserve(trk.observations.size());
    for (const auto& kv : trk.observations) obs_frames.push_back(kv.first);
    if (obs_frames.empty()) continue;

    // 刚性静态剪枝
    if (params_.min_rigid_disp > 0.0 && (int64_t)obs_frames.size() >= params_.min_hits_prune) {
      if (obs_frames.size() > 1) {
        std::vector<float> xs, ys;
        xs.reserve(obs_frames.size());
        ys.reserve(obs_frames.size());
        for (int64_t f : obs_frames) {
          xs.push_back(trk.observations.find(f)->second.pos[0]);
          ys.push_back(trk.observations.find(f)->second.pos[1]);
        }
        const double net_disp =
            (double)NpNorm1Df(xs.back() - xs.front(), ys.back() - ys.front());
        const float var_x = NpVarF(xs);
        const float var_y = NpVarF(ys);
        const float pos_var_f = var_x + var_y;  // numpy: float32 + float32
        const double pos_var = (double)pos_var_f;
        const double centroid_y = (double)NpMeanF(ys);
        // numpy: self.sky_y_boundary 是 python float，centroid_y 也是 python float
        //        ⇒ 这一处比较是 float64（与 Step D 里的 float32 比较不同）
        const double sky_boundary_d = (double)params_.img_h * params_.sky_ratio;
        double effective_rigid_disp = params_.min_rigid_disp;
        const bool has_sky_override = params_.min_rigid_disp_sky >= 0.0;
        if (has_sky_override && centroid_y < sky_boundary_d) {
          effective_rigid_disp = params_.min_rigid_disp_sky;
        }
        if (effective_rigid_disp > 0.0) {
          if (net_disp < effective_rigid_disp && pos_var < params_.max_rigid_var) continue;
        } else if (centroid_y < sky_boundary_d) {
          // 天空轨迹豁免
        } else {
          if (net_disp < params_.min_rigid_disp && pos_var < params_.max_rigid_var) continue;
        }
      }
    }

    const bool can_infill = trk.hits >= params_.min_hits_infill;

    // 1) 输出全部真实观测（含 warmup 前缀）
    for (int64_t f : obs_frames) {
      if (f < 0 || f >= num_frames) continue;
      const Observation& o = trk.observations.find(f)->second;
      TrackFusionDet d;
      d.track_id = trk.id;
      d.x = o.pos[0];
      d.y = o.pos[1];
      d.score = o.score;
      d.infilled = !o.measured;
      frame_outputs[(size_t)f].push_back(d);
    }

    // 2) 内部空洞插补（legacy 口径：gap <= infill_gap + 1，dedup_radius = 0）
    if (can_infill && obs_frames.size() >= 2) {
      for (size_t i = 0; i + 1 < obs_frames.size(); ++i) {
        const int64_t f1 = obs_frames[i];
        const int64_t f2 = obs_frames[i + 1];
        const int64_t gap = f2 - f1;
        if (gap <= 1) continue;
        const bool is_legacy_gap = gap <= (params_.infill_gap + 1);
        if (!is_legacy_gap) continue;
        const Observation& o1 = trk.observations.find(f1)->second;
        const Observation& o2 = trk.observations.find(f2)->second;
        for (int64_t step = 1, missing = f1 + 1; missing < f2; ++step, ++missing) {
          const double alpha = (double)step / (double)gap;
          const float a32 = (float)(1.0 - alpha);
          const float b32 = (float)alpha;
          const float ix = a32 * o1.pos[0] + b32 * o2.pos[0];
          const float iy = a32 * o1.pos[1] + b32 * o2.pos[1];
          const double iscore = (1.0 - alpha) * o1.score + alpha * o2.score;
          if (missing < 0 || missing >= num_frames) continue;
          TrackFusionDet d;
          d.track_id = trk.id;
          d.x = ix;
          d.y = iy;
          d.score = iscore;
          d.infilled = true;
          frame_outputs[(size_t)missing].push_back(d);
        }
      }
    }
  }

  // ---- Pass 3: 连续帧确认 ----
  const int32_t confirm_consec = std::max(1, params_.confirm_consec);
  const int32_t confirm_gap_tol = std::max(0, params_.confirm_gap_tol);
  const int64_t max_gap = 1 + confirm_gap_tol;

  std::map<int32_t, std::vector<int64_t>> observed;
  for (size_t f = 0; f < frame_outputs.size(); ++f) {
    for (const TrackFusionDet& d : frame_outputs[f]) {
      if (!d.infilled) observed[d.track_id].push_back((int64_t)f);
    }
  }
  std::map<int32_t, int64_t> confirmed_at;
  for (auto& kv : observed) {
    std::vector<int64_t> frames = kv.second;
    std::sort(frames.begin(), frames.end());
    frames.erase(std::unique(frames.begin(), frames.end()), frames.end());
    // 切成 run
    std::vector<std::pair<int64_t, int64_t>> runs;  // (length, start)
    int64_t run_start = frames[0];
    int64_t run_length = 1;
    int64_t previous = frames[0];
    for (size_t i = 1; i < frames.size(); ++i) {
      const int64_t frame = frames[i];
      if (frame - previous <= max_gap) {
        run_length += 1;
      } else {
        runs.emplace_back(run_length, run_start);
        run_start = frame;
        run_length = 1;
      }
      previous = frame;
    }
    runs.emplace_back(run_length, run_start);
    int64_t best = 0;
    for (const auto& r : runs) best = std::max(best, r.first);
    if (best < confirm_consec) continue;  // 闪断目标整轨丢弃
    for (const auto& r : runs) {
      if (r.first >= confirm_consec) {
        confirmed_at[kv.first] = r.second;
        break;
      }
    }
  }

  // 过滤 + 按分数降序（Python 的 list.sort 是稳定排序）
  std::vector<std::vector<TrackFusionDet>> filtered(frame_outputs.size());
  for (size_t f = 0; f < frame_outputs.size(); ++f) {
    std::vector<TrackFusionDet>& kept = filtered[f];
    for (const TrackFusionDet& d : frame_outputs[f]) {
      auto it = confirmed_at.find(d.track_id);
      if (it == confirmed_at.end()) continue;
      if (params_.confirm_trim_prefix && (int64_t)f < it->second) continue;
      kept.push_back(d);
    }
    std::stable_sort(kept.begin(), kept.end(),
                     [](const TrackFusionDet& a, const TrackFusionDet& b) { return a.score > b.score; });
  }

  fused_ = filtered;
  *out = std::move(filtered);
}

}  // namespace mosaic
