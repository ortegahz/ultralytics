// ===========================================================================
// mosaic_track_fusion.h -- 时空融合（spatio-temporal track fusion）后处理
//
//   这是 Python 参考实现「osd_sys_sota_trial0080_th045_20261022」的那一套
//   时空逻辑的 **C++ 移植**（交付给嵌入式组的 SDK 内核）：
//
//     manu/evaluation/eval_bidirectional_track_fusion.py
//       Pass 1  OnlineAdaptiveTracker   在线自适应两档打捞 + 卡尔曼关联
//       Pass 2  BidirectionalTemporalSmoother
//               stitch_tracklets        碎片缝合
//               smooth_and_infill       双向平滑 + 内部空洞插补 + 刚性坏点剪枝
//     manu/evaluation/system_sota_longquan.py
//       Pass 3  apply_consecutive_confirmation  连续帧确认 + 前缀裁剪 + 按分数排序
//
//   设计约束（**不要破坏**）：
//     1) 本文件**零第三方依赖**（只用 C++ 标准库）⇒ 能在 host 上单独编译做
//        精度对齐，也能在板卡上原生编译，不受 OpenCV / RKNN / RGA 版本影响。
//     2) 所有数值语义严格照抄 numpy：
//        - numpy 的 float32 数组运算按 float32 走；
//        - **numpy 标量与 Python float 比较时按 float32 比**（NEP 50：Python
//          标量是「弱类型」，跟随数组 dtype），所以阈值比较一律先把阈值降到
//          float32；
//        - `np.linalg.norm` 在 float32 上是 sqrt(add.reduce(x*x))，即 float32
//          乘法 + float32 求和 + float32 开方；
//        - 卡尔曼的矩阵乘按 float32 存回（内部用 double 累加以贴近 BLAS 的
//          FMA 结果）。
//     3) 匈牙利关联必须与 scipy.optimize.linear_sum_assignment **逐位一致**
//        ⇒ 这里移植的是 scipy 1.15.3 内部用的 Crouse 最短增广路算法
//        （IEEE TAES 52(4):1679-1796, 2016），代价矩阵先上转 float64。
// ===========================================================================
#ifndef MOSAIC_TRACK_FUSION_H_
#define MOSAIC_TRACK_FUSION_H_

#include <cstdint>
#include <map>
#include <vector>

namespace mosaic {

// ---------------------------------------------------------------------------
// 参数。字段名与 Python 的 params 字典**一一对应**。
//
// 🔴 默认值 = **2026-11-20 交付组**（trial_0080 底座 + v3 工作点上的 Δn 重搜，
//     交付档 manu/evaluation/system_sota_longquan_trial0080_delivery_v3_optgapfit_params.json，
//     params_md5 a1157e013ebb496731be805c72c95dd8）。
//     这不是 th045/v1 冻结档了 —— 阈值也已经从 legacy 迁到 v3 工作点。
// ---------------------------------------------------------------------------
struct TrackFusionParams {
  // ---- 阈值与天地分区 ----
  double th_base = 0.48;        // 天空区高置信阈值（v3 工作点；legacy th045 是 0.45）
  double th_salvage = 0.06;     // 天空区弱脉冲打捞阈值
  double th_ground = 0.26;      // 地面区阈值（v3 工作点；legacy th045 是 0.22）
  double sky_ratio = 1.00;      // 天空/地面分界 = img_h * sky_ratio
                               //   v3: 1.0 ⇒ boundary = 512 ⇒ is_sky 恒真（与 Python v3 同语义）
  int32_t img_h = 512;          // 帧高（Longquan 原生 640x512 ⇒ 512）

  // ---- Pass 1: 在线跟踪器 ----
  int32_t max_age = 3;          // 失配多少帧后判死
  int32_t min_hits = 3;         // 轨迹确认所需命中数
  double match_dist = 10.0;     // 关联基础门限（交付组；旧搜索产物是 16.0）
  double max_match_dist = 24.0; // 关联门限上限
  double min_track_score = 0.08;// 轨迹分数下限
  double instant_conf = 0.25;   // 新建轨迹的「立即确认」阈值
  double min_displacement = 2.5;// 地面静态杂波剔除的最小净位移
  double th_deep = 0.0;         // 深潜打捞阈值（0 = 关闭）
  int32_t min_hits_deep = 3;    // 深潜打捞所需的最小命中数

  // ---- Pass 1: 物理帧间隔（Δn）感知 ----
  //     检测器按「源帧号」跳着送更新（板载 ~8fps / 源 25fps ⇒ Δn 均值 3.11），
  //     所以引擎要按**物理帧间隔**补偿：卡尔曼按 dt 步进、关联门限按 dt 放宽、
  //     轨迹分数按 dt 衰减。
  //
  //     这三个默认值来自 2026-11-20 的**交付底座重搜**
  //     （trial_0080 重建缓存 + v3 工作点，24 序 / 165,223 帧全口径）：
  //     max_age_phys=-1（**关闭** ⇒ 走 legacy 按更新次数判死）、r_slope=5.0 px/帧、
  //     alpha_decay=1.0（实测不需要额外衰减）。
  //     ⚠️ max_age_phys 交付值是「不设」⇒ 必须是 **负数**，不是 0
  //        （写 0 会让航迹下一帧就死）。旧搜索产物的 10（= 400 ms 物理帧寿命）
  //        在全口径上 FAR 0.0352 ⇒ 破 0.03 红线，已被交付闸门拒绝。
  int32_t max_age_phys = -1;    // 物理帧龄上限（<0 => legacy：按更新次数判死）
  double r_slope = 5.0;         // 每多一个物理帧，关联门限放宽的像素数
  double alpha_decay = 1.0;     // 每个物理帧的分数保持率

  // ---- Pass 2: 缝合 / 插补 / 刚性剪枝 ----
  int32_t stitch_gap = 6;       // 允许缝合的最大帧间隔
  double stitch_max_dist = 25.0;// 缝合的空间容差下限
  int32_t infill_gap = 3;       // 允许插补的最大内部空洞
  int32_t min_hits_infill = 5;  // 允许插补所需的最小命中数
  double min_rigid_disp = 2.0;  // 刚性坏点剪枝的最小净位移
  double max_rigid_var = 1.0;   // 刚性坏点剪枝的最大坐标方差
  int32_t min_hits_prune = 4;   // 启动刚性剪枝所需的最小观测帧数
  double min_rigid_disp_sky = -1.0;  // <0 ⇒ None ⇒ 走 legacy 分支

  // ---- Pass 3: 连续帧确认 ----
  //
  //     🔴 这两个默认值来自 2026-11-20 的交付底座重搜（v3 工作点 + 跳帧工况）：
  //     Δn ≈ 3.1 时「必须逐帧连续」的判定（confirm_gap_tol=0 ⇒ max_gap=1 ⇒
  //     物理帧差 >1 就断 run）会把**所有**轨迹判死（v3 原参在跳帧臂 F1 = 0.084122）。
  //     交付组是「2 次真实观测 + 容忍 8 个物理帧的间隔」。
  //     ⚠️ 这里的帧差必须是**物理源帧号之差**，不是「距上次更新的次数」；
  //        且 infilled（合成）帧**不计入** confirm_consec。
  //     冻结的 th045/v1 基线是 confirm_consec=6 / confirm_gap_tol=0。
  int32_t confirm_consec = 2;      // 连续命中多少帧才算确认
  int32_t confirm_gap_tol = 8;     // 连续判定允许的额外物理帧间隔
  int32_t confirm_trim_prefix = 1; // 是否裁掉确认帧之前的检测
};

// 一条输入候选（原生帧像素坐标 + 热图峰值）。
struct TrackFusionCand {
  float x = 0.0f;
  float y = 0.0f;
  float score = 0.0f;
};

// 一条输出检测。score 用 double：Python 侧的分数是 float64（合成帧的插值
// 分数更是纯 Python float），用 float 会丢精度。
struct TrackFusionDet {
  int32_t track_id = 0;
  float x = 0.0f;
  float y = 0.0f;
  double score = 0.0;
  bool infilled = false;  // true = 合成（插补）帧，不是真实观测
};

// ---------------------------------------------------------------------------
// 时空融合引擎。
//
// 生命周期与 Python 参考实现一一对应：
//   for each frame:  engine.Step(frame_idx, cands, n);      // Pass 1
//   after the last:  engine.Finalize(&frames);              // Pass 2 + 3
//
// Finalize() 返回 **每个原始帧** 一组检测（含插补出来的合成帧），并按
// 「分数降序」排好 —— 与 apply_consecutive_confirmation 的输出顺序一致。
// ---------------------------------------------------------------------------
class TrackFusionEngine {
 public:
  explicit TrackFusionEngine(const TrackFusionParams& params);

  void Reset();

  // Pass 1：喂入一帧的候选池（天地分区 + 双层阈值 + 卡尔曼关联）。
  void Step(int64_t frame_idx, const TrackFusionCand* cands, size_t count);

  // Pass 2 + Pass 3：缝合 / 插补 / 刚性剪枝 / 连续帧确认。
  void Finalize(std::vector<std::vector<TrackFusionDet>>* out);

  int64_t num_frames() const { return num_frames_; }
  const TrackFusionParams& params() const { return params_; }

  // 「在线融合」输出：只含**真实观测**（非合成帧）、且当时已确认的轨迹。
  // 用于 RT pipeline 的逐帧在线消费（参考实现的 online_fusion 口径）。
  void OnlineFused(int64_t frame_idx, std::vector<TrackFusionDet>* out) const;

 private:
  struct Observation {
    float pos[2] = {0.0f, 0.0f};
    double score = 0.0;
    bool measured = true;
  };

  struct Track {
    int32_t id = 0;
    float x[4] = {0.0f, 0.0f, 0.0f, 0.0f};   // [x, y, vx, vy]
    float P[4][4] = {{0.0f}};
    float Q[4][4] = {{0.0f}};
    float R[2][2] = {{0.0f}};
    float F[4][4] = {{0.0f}};
    float H[2][4] = {{0.0f}};
    int64_t start_frame = 0;
    int64_t last_frame = 0;
    int64_t last_observed_frame = 0;
    int64_t last_hit_fid = 0;   // 最近一次**命中**（Update）的物理帧号
    int32_t hits = 1;
    int32_t age = 1;
    int32_t time_since_update = 0;
    double score = 0.0;
    double init_score = 0.0;
    bool is_confirmed = false;
    float start_pos[2] = {0.0f, 0.0f};
    std::map<int64_t, Observation> observations;
  };

  void InitTrackMatrices(Track* t) const;
  void Predict(Track* t, int64_t frame_idx, int64_t dt) const;
  void Update(Track* t, const float pos[2], double score, int64_t frame_idx) const;
  float VelocityNorm(const Track& t) const;
  double NetDisplacement(const Track& t) const;
  bool IsConfirmed(const Track& t) const;

  TrackFusionParams params_;
  int32_t next_track_id_ = 1;
  int64_t num_frames_ = 0;
  int64_t last_step_frame_ = -1;  // 上一次**被接受**的 Step 的物理帧号（-1 = 无）
  std::vector<Track> active_;
  std::vector<Track> finished_;
  std::vector<Track> stitched_;
  std::vector<std::vector<TrackFusionDet>> fused_;
};

// ---------------------------------------------------------------------------
// 独立工具：矩形线性指派（scipy.optimize.linear_sum_assignment 的等价实现）。
//
// 移植自 scipy 1.15.3 内部的 Crouse 最短增广路算法，代价矩阵先上转 float64。
// 返回配对数量 = min(nrow, ncol)；row/col 数组按行序（转置情形按列序）给出。
// ---------------------------------------------------------------------------
void LinearSumAssignment(const double* cost, int nrow, int ncol, std::vector<int32_t>* row_ind,
                         std::vector<int32_t>* col_ind);

}  // namespace mosaic

#endif  // MOSAIC_TRACK_FUSION_H_
