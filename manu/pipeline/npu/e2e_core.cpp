// =============================================================================
// e2e_core.cpp -- implementation of the two reusable e2e stages
//
// This file does NOT reimplement anything. It #includes the two production
// sources with their main() compiled out and calls the identical classes and
// functions they use:
//
//   stage A:  OnlineFeaturePipeline / FusedGpu / find_sequence_folder / list_dir
//             (from ../cpp/gmc_stream_ocl.cpp)
//   stage B:  letterbox / run_once / dequant / extract_peaks / Ctx / read_file
//             (from rknn_heatmap_smoke.cpp)
//
// Both anonymous namespaces merge into this one TU. The two files declare no
// colliding top-level names (verified by comm over their definition lists), so
// the merge is safe and no macro renaming is needed.
// =============================================================================

#include "e2e_core.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

// Plain NEON (uint8->float32 widening) for u8_to_f32 below. vcvtq_f32_u32 is
// baseline aarch64; the fp16 intrinsics arm_neon.h also declares are NOT, which
// is why f32_to_f16_bits is a bits-level conversion.
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#include <sys/stat.h>
#include <sys/types.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

// ---- production cores, main() compiled out ---------------------------------
#define GMC_STREAM_OCL_NO_MAIN
#include "../cpp/gmc_stream_ocl.cpp"

#define RKNN_HEATMAP_SMOKE_NO_MAIN
#include "rknn_heatmap_smoke.cpp"

namespace e2ecore {

// =============================================================================
// Streaming implementations -- these hold THE per-frame bodies.
// The batch entry points below are thin loops over these, so the serial demo
// and a pipelined demo cannot drift: there is only one copy of the code.
// =============================================================================
struct FeatureStream {
    std::string sequence, out_dir, out_ext;
    std::vector<std::string> names;
    size_t idx = 0;
    // gpu MUST be declared before pipe: pipe holds a raw pointer to it, so it
    // has to outlive the pipeline, and members destroy in reverse declaration
    // order (pipe first, then gpu).
    std::unique_ptr<FusedGpu> gpu;
    gmcpp::MD5 md5;
    OnlineFeaturePipeline pipe;

    FeatureStream(const FeatureParams& p, std::vector<std::string> names_in,
                  OnlineFeaturePipeline::Backend backend)
        : sequence(p.sequence), out_dir(p.out_dir), out_ext(p.out_ext),
          names(std::move(names_in)),
          gpu(backend == OnlineFeaturePipeline::kCpu
                  ? nullptr
                  : new FusedGpu(p.kernel_dir, false, false, p.window * p.stride_step + 3)),
          pipe(p.window, p.stride_step, p.anchor_step, p.downscale,
               /*rng_seed_per_fit=*/false, p.mode == "nogmc", gpu.get(), backend,
               /*pyr_cache=*/false, p.fit_corners, p.fit_win, p.fit_level, p.fit_iters,
               p.cache_parts, p.threads) {
        pipe.set_profiling(false);   // a timing ledger only; it cannot move pixels
        pipe.set_seq_id(0);
        if (p.gate_shift > 0.0) pipe.set_fit_gate(p.gate_shift, p.gate_min, p.gate_max);
        pipe.gpu_prealloc(0);
        pipe.gpu_pin_pad(false);
        pipe.set_corner_probe(p.corner_probe != 0);
        pipe.set_fit_deadline(0.0, false);   // deterministic: output independent of load
    }
};

struct InferEngine {
    Ctx c;
    std::string dump_dir, detections_out, tag, input_dump_dir;
    float th = 0.25f;
    int stride = 2, top_k = 100;
    bool dump_input = false;
    bool want_perf = false;
    int64_t last_run_us = 0;   // PERF_RUN is CUMULATIVE since rknn_init, not per frame
    std::ofstream detf;
    std::vector<uint8_t> input;
    std::vector<float> hm, off;
    std::vector<rknn_output> out;
    std::vector<Det> dets;
    long frames = 0, detections = 0;
    bool want_hash = false;
    bool use_io_mem = false;
    float conv_div = 1.0f;
    uint32_t hash = 2166136261u;   // FNV-1a offset basis
    InferTiming acc;               // cumulative sub-stage totals
};

FeatureStream* feature_stream_open(const FeatureParams& p) {
    std::string why;
    if (!gmcpp::md5_self_test(&why)) {
        std::fprintf(stderr, "[FATAL] built-in md5 self-test failed (%s)\n", why.c_str());
        return nullptr;
    }
    std::string seq_dir;
    if (!find_sequence_folder(p.raw_root, p.sequence, &seq_dir)) {
        std::fprintf(stderr, "[FATAL] sequence folder not found for %s under %s\n",
                     p.sequence.c_str(), p.raw_root.c_str());
        return nullptr;
    }
    std::vector<std::string> names = list_dir(seq_dir, true);
    std::sort(names.begin(), names.end(), natural_less);
    if (p.limit > 0 && static_cast<int>(names.size()) > p.limit)
        names.resize(static_cast<size_t>(p.limit));
    if (names.empty()) {
        std::fprintf(stderr, "[FATAL] no frames in %s\n", seq_dir.c_str());
        return nullptr;
    }

    OnlineFeaturePipeline::Backend backend = OnlineFeaturePipeline::kCpu;
    if (p.fused == "cpu")       backend = OnlineFeaturePipeline::kCpu;
    else if (p.fused == "gpu")  backend = OnlineFeaturePipeline::kGpu;
    else if (p.fused == "both") backend = OnlineFeaturePipeline::kBoth;
    else {
        std::fprintf(stderr, "[FATAL] fused must be 'cpu', 'gpu' or 'both', got '%s'\n",
                     p.fused.c_str());
        return nullptr;
    }

    // Carry the FULL path on the frame list so feature_stream_next() never has
    // to re-derive it. list_dir() returns bare names, and a bare name silently
    // reads the wrong file (or nothing) once the process is no longer sitting in
    // the sequence directory.
    for (size_t i = 0; i < names.size(); ++i) names[i] = seq_dir + "/" + names[i];

    FeatureStream* s;
    try {
        s = new FeatureStream(p, names, backend);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FATAL] feature pipeline construction failed: %s\n", e.what());
        return nullptr;
    }
    if (backend != OnlineFeaturePipeline::kCpu)
        std::printf("[BUILD] backend=%s anchor_step=%d opencv=%s\n", p.fused.c_str(), p.anchor_step,
                    CV_VERSION);
    if (!p.out_dir.empty()) mkdir_p(p.out_dir);

    const std::string aff = bind_cpus(parse_cpu_list("none"));
    std::fprintf(stderr, "[THREADS]   requested=%d affinity=%s (effective opencv threads=%d)\n",
                 p.threads, aff.c_str(), cv::getNumThreads());

    std::printf("[FIT-GATE] shift<=%.2f x frame  scale in [%.2f,%.2f]\n", p.gate_shift,
                p.gate_min, p.gate_max);
    std::printf("ring_depth=%ld lags=%d..%d\n", s->pipe.ring_depth(), s->pipe.lags().front(),
                s->pipe.lags().back());
    std::printf("[BUILD] window=%d stride_step=%d anchor_step=%d downscale=%d mode=%s opencv=%s\n",
                p.window, p.stride_step, p.anchor_step, p.downscale, p.mode.c_str(), CV_VERSION);
    return s;
}

bool feature_stream_next(FeatureStream* s, std::string* out_bmp, FrameTiming* timing) {
    if (s->idx >= s->names.size()) return false;
    const size_t fi = s->idx;
    const std::string& path = s->names[fi];

    s->pipe.set_seq_frame(static_cast<int>(fi));
    const auto& st0 = s->pipe.stats();
    const double b_fit = st0.t_fit, b_warp = st0.t_warp, b_med = st0.t_median,
                 b_tot = st0.t_total;
    const long b_np = st0.n_push;

    const Mat frame = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (frame.empty()) {
        std::fprintf(stderr, "[FATAL] failed to read %s\n", path.c_str());
        return false;
    }
    const Mat out = s->pipe.push(frame, nullptr);

    const auto& st1 = s->pipe.stats();
    if (timing) {
        timing->fit_ms = st1.t_fit - b_fit;
        timing->warp_ms = st1.t_warp - b_warp;
        timing->median_ms = st1.t_median - b_med;
        timing->total_ms = st1.n_push > b_np ? st1.t_total - b_tot : 0.0;
    }

    gmcpp::MD5 f;
    f.update(reinterpret_cast<const uint8_t*>(out.data),
             static_cast<size_t>(out.total()));
    s->md5.update(f.hex());

    Mat hwc(frame.rows, frame.cols, CV_8UC3);
    for (int y = 0; y < frame.rows; ++y)
        for (int x = 0; x < frame.cols; ++x) {
            const int q = y * frame.cols + x;
            hwc.at<cv::Vec3b>(y, x)[0] = out.at<uint8_t>(0, q);
            hwc.at<cv::Vec3b>(y, x)[1] = out.at<uint8_t>(1, q);
            hwc.at<cv::Vec3b>(y, x)[2] = out.at<uint8_t>(2, q);
        }
    char nm[256];
    std::snprintf(nm, sizeof(nm), "%s_%06zu.%s", s->sequence.c_str(), fi, s->out_ext.c_str());
    const std::string full = s->out_dir + "/" + nm;
    if (!cv::imwrite(full, hwc)) {
        std::fprintf(stderr, "[FATAL] failed to write %s\n", full.c_str());
        return false;
    }
    *out_bmp = full;
    ++s->idx;
    return true;
}

// Produces the frame straight into an RGB buffer. Deliberately shares the
// push/md5 work with feature_stream_next() via a small helper so the two
// cannot drift on the parts that decide the feature VALUES.
bool feature_stream_next_rgb(FeatureStream* s, std::vector<uint8_t>* rgb, int* w, int* h,
                             FrameTiming* timing) {
    if (s->idx >= s->names.size()) return false;
    const size_t fi = s->idx;
    const std::string& path = s->names[fi];
    s->pipe.set_seq_frame(static_cast<int>(fi));
    const auto& st0 = s->pipe.stats();
    const double b_fit = st0.t_fit, b_warp = st0.t_warp, b_med = st0.t_median, b_tot = st0.t_total;
    const long b_np = st0.n_push;

    const Mat frame = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (frame.empty()) {
        std::fprintf(stderr, "[FATAL] failed to read %s\n", path.c_str());
        return false;
    }
    const Mat out = s->pipe.push(frame, nullptr);
    const auto& st1 = s->pipe.stats();
    if (timing) {
        timing->fit_ms = st1.t_fit - b_fit;
        timing->warp_ms = st1.t_warp - b_warp;
        timing->median_ms = st1.t_median - b_med;
        timing->total_ms = st1.n_push > b_np ? st1.t_total - b_tot : 0.0;
    }
    gmcpp::MD5 f;
    f.update(reinterpret_cast<const uint8_t*>(out.data), static_cast<size_t>(out.total()));
    s->md5.update(f.hex());

    const int W = frame.cols, H = frame.rows;
    const size_t plane = static_cast<size_t>(W) * static_cast<size_t>(H);
    rgb->assign(plane * 3, 0);
    const uint8_t* c0 = out.ptr<uint8_t>(0);
    const uint8_t* c1 = out.ptr<uint8_t>(1);
    const uint8_t* c2 = out.ptr<uint8_t>(2);
    for (size_t p = 0; p < plane; ++p) {           // RGB order == stb_image's read of the BMP
        (*rgb)[p * 3 + 0] = c2[p];
        (*rgb)[p * 3 + 1] = c1[p];
        (*rgb)[p * 3 + 2] = c0[p];
    }
    *w = W; *h = H;
    ++s->idx;
    return true;
}

std::string feature_stream_md5(FeatureStream* s) { return s->md5.hex(); }
long feature_stream_remaining(FeatureStream* s) {
    return static_cast<long>(s->names.size() - s->idx);
}
void feature_stream_handles(FeatureStream* s, long* fitgate, long* chain,
                            long* fit_deadline, long* corner_cap) {
    const auto& st = s->pipe.stats();
    if (fitgate)      *fitgate = st.n_fit_rejected;
    if (chain)        *chain = s->pipe.n_chain_rejected();
    if (fit_deadline) *fit_deadline = s->pipe.n_fit_deadline();
    if (corner_cap)   *corner_cap = s->pipe.corner_cap_frames();
}
void feature_stream_close(FeatureStream* s) {
    // Plain delete only. Calling s->pipe.~OnlineFeaturePipeline() here as well
    // destroys the pipeline twice -- members are destroyed by ~FeatureStream in
    // reverse declaration order -- which shows up as
    // "free(): double free detected in tcache 2", typically hundreds of frames
    // in and only when stdout happens to be a file.
    delete s;
}

InferEngine* infer_open(const InferParams& p) {
    std::vector<uint8_t> model;
    if (!read_file(p.model_path.c_str(), &model)) {
        std::fprintf(stderr, "[FATAL] cannot read model %s\n", p.model_path.c_str());
        return nullptr;
    }
    std::printf("[1] model: %zu bytes, fnv1a32=%08x\n", model.size(),
                fnv1a(model.data(), model.size()));

    InferEngine* e = new InferEngine();
    e->dump_dir = p.dump_dir;
    e->detections_out = p.detections_out;
    e->tag = p.tag;
    e->input_dump_dir = p.input_dump_dir;
    e->th = p.th;
    e->stride = p.stride;
    e->top_k = p.top_k;
    e->dump_input = p.dump_input;
    e->want_perf = p.rknn_perf;

    const uint32_t rknn_flag = p.rknn_perf ? RKNN_FLAG_COLLECT_PERF_MASK : 0u;
    if (rknn_init(&e->c.ctx, model.data(), (uint32_t)model.size(), rknn_flag, NULL) != 0) {
        std::fprintf(stderr, "[FATAL] rknn_init failed\n");
        delete e;
        return nullptr;
    }
    std::printf("[2] rknn_init ok\n");

    rknn_input_output_num io;
    std::memset(&io, 0, sizeof(io));
    rknn_query(e->c.ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    if (io.n_input != 1 || io.n_output != 2) {
        std::fprintf(stderr, "[FATAL] expected 1 input / 2 outputs (heatmap, offset)\n");
        rknn_destroy(e->c.ctx);
        delete e;
        return nullptr;
    }
    rknn_tensor_attr in_attr;
    std::memset(&in_attr, 0, sizeof(in_attr));
    in_attr.index = 0;
    rknn_query(e->c.ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
    if (in_attr.fmt == RKNN_TENSOR_NCHW) {
        e->c.in_c = (int)in_attr.dims[1]; e->c.in_h = (int)in_attr.dims[2];
        e->c.in_w = (int)in_attr.dims[3];
    } else {
        e->c.in_h = (int)in_attr.dims[1]; e->c.in_w = (int)in_attr.dims[2];
        e->c.in_c = (int)in_attr.dims[3];
    }
    // Remember whether the graph wants FLOAT16 input. Only then is handing
    // rknn_inputs_set a pre-widened buffer legal; on an INT8 graph the runtime
    // quantises whatever it is given and the uint8 path stays the right one.
    e->c.input_is_f16 = (in_attr.type == RKNN_TENSOR_FLOAT16);
    e->c.input_is_int8 = (in_attr.type == RKNN_TENSOR_INT8);
    e->c.input_zp = in_attr.zp;
    e->use_io_mem = p.use_io_mem;
    e->conv_div = p.conv_div;
    if (e->use_io_mem) {
        // Zero-copy input: the host owns the input tensor's storage and
        // rknn_inputs_set is never called. rknn_set_io_mem has no driver-side
        // size check, which is exactly what pass_through lacks here (the
        // driver wants 4 B/px while this graph declares 2 B/px).
        const uint32_t mem_sz = in_attr.size_with_stride ? in_attr.size_with_stride : in_attr.size;
        e->c.io_mem = rknn_create_mem(e->c.ctx, mem_sz);
        if (!e->c.io_mem) {
            std::fprintf(stderr, "[FATAL] rknn_create_mem(%u) failed\n", mem_sz);
            rknn_destroy(e->c.ctx);
            delete e;
            return nullptr;
        }
        if (rknn_set_io_mem(e->c.ctx, e->c.io_mem, &in_attr) != 0) {
            std::fprintf(stderr, "[FATAL] rknn_set_io_mem failed\n");
            rknn_destroy(e->c.ctx);
            delete e;
            return nullptr;
        }
        std::printf("[2] io-mem: zero-copy input bound (%u B, type=%d zp=%d conv_div=%.3f)\n",
                    mem_sz, (int)in_attr.type, in_attr.zp, e->conv_div);
    }
    if (e->c.input_is_f16)
        std::printf("[2] model input type = FLOAT16 -> pre-widened input path available\n");
    // The declared size is printed because it is the only way to settle a
    // disagreement the runtime raised: on the fp16 graph it refuses a 2457600 B
    // upload with "param input size(2457600) < model input size(4915200)". If the
    // attr itself says 2457600, then the 4915200 in that message does not come
    // from this model's declared input and pass_through cannot be reasoned about
    // from the header -- it has to be treated as a runtime bug, not a caller error.
    std::printf("[2] input attr: type=%d fmt=%d n_elems=%u size=%u dims=[%u %u %u %u]\n",
                (int)in_attr.type, (int)in_attr.fmt, in_attr.n_elems, in_attr.size,
                in_attr.dims[0], in_attr.dims[1], in_attr.dims[2], in_attr.dims[3]);
    if (e->c.in_h != kDefaultSize || e->c.in_w != kDefaultSize || e->c.in_c != 3) {
        std::fprintf(stderr, "[FATAL] model input is %dx%dx%d, expected 640x640x3\n",
                     e->c.in_h, e->c.in_w, e->c.in_c);
        rknn_destroy(e->c.ctx);
        delete e;
        return nullptr;
    }
    e->c.n_out = io.n_output;
    e->c.out_attr.assign(e->c.n_out, rknn_tensor_attr{});
    for (uint32_t i = 0; i < e->c.n_out; i++) {
        std::memset(&e->c.out_attr[i], 0, sizeof(rknn_tensor_attr));
        e->c.out_attr[i].index = i;
        rknn_query(e->c.ctx, RKNN_QUERY_OUTPUT_ATTR, &e->c.out_attr[i], sizeof(e->c.out_attr[i]));
    }
    {
        const rknn_tensor_attr& a = e->c.out_attr[0];
        if (a.n_dims != 4 || a.dims[0] != 1 || a.dims[1] != 1) { rknn_destroy(e->c.ctx); delete e; return nullptr; }
        e->c.hm_h = (int)a.dims[2]; e->c.hm_w = (int)a.dims[3];
        const rknn_tensor_attr& b = e->c.out_attr[1];
        if (b.n_dims != 4 || b.dims[0] != 1 || b.dims[1] != 2) { rknn_destroy(e->c.ctx); delete e; return nullptr; }
        e->c.off_h = (int)b.dims[2]; e->c.off_w = (int)b.dims[3];
        if (e->c.off_h != e->c.hm_h || e->c.off_w != e->c.hm_w ||
            e->c.in_h / e->c.hm_h != e->stride) {
            std::fprintf(stderr, "[FATAL] grid/stride mismatch\n");
            rknn_destroy(e->c.ctx);
            delete e;
            return nullptr;
        }
        std::printf("    heatmap %dx%d, offset %dx%d, implied stride %d\n", e->c.hm_h, e->c.hm_w,
                    e->c.off_h, e->c.off_w, e->c.in_h / e->c.hm_h);
    }

    if (!e->dump_dir.empty()) mkdir_p(e->dump_dir.c_str());
    if (!e->detections_out.empty()) {
        e->detf.open(e->detections_out, std::ios::out | std::ios::trunc);
        if (!e->detf) {
            std::fprintf(stderr, "[FATAL] cannot open %s\n", e->detections_out.c_str());
            rknn_destroy(e->c.ctx);
            delete e;
            return nullptr;
        }
    }
    std::printf("[3] infer engine ready  th=%.4f stride=%d topk=%d\n", e->th, e->stride, e->top_k);
    return e;
}

// THE per-frame body. Both the file path and the no-I/O path land here, so they
// cannot drift; the only difference is who supplies the RGB bytes and whether
// anything gets written to disk.
static bool infer_frame(InferEngine* e, const uint8_t* rgb, int sw, int sh,
                        const uint8_t* preboxed, const char* out_name, long* n_det_out,
                        InferTiming* t, const uint16_t* preboxed_f16 = nullptr,
                        const float* preboxed_f32 = nullptr,
                        bool f32_passthrough = true) {
    const double t_frame0 = now_ms();
    if (preboxed) {
        // Letterbox already ran on its own stage; copy in without redoing it.
        e->input.assign(preboxed, preboxed + (size_t)kDefaultSize * kDefaultSize * 3);
        if (t) t->letterbox_ms = 0.0;
    } else {
        const double t_lb0 = now_ms();
        letterbox(rgb, sw, sh, kDefaultSize, &e->input);
        if (t) t->letterbox_ms = now_ms() - t_lb0;
    }

    int mx = 0;
    for (uint8_t v : e->input) mx = std::max(mx, (int)v);
    if (mx == 0) {
        std::fprintf(stderr, "[FATAL] letterboxed input is all zero: %s\n", out_name);
        return false;
    }

    if (e->dump_input && !e->input_dump_dir.empty()) {
        const char* slash = std::strrchr(out_name, '/');
        std::string stem_name = slash ? slash + 1 : out_name;
        for (auto& ch : stem_name) if (ch == '.') ch = '_';
        char buf[1200];
        std::snprintf(buf, sizeof(buf), "%s%s%s.input.bin", e->input_dump_dir.c_str(),
                      e->input_dump_dir.back() == '/' ? "" : "/", stem_name.c_str());
        if (mkdir(e->input_dump_dir.c_str(), 0755) != 0 && errno != EEXIST) { /* exists */ }
        write_bin(buf, e->input.data(), e->input.size());
    }

    double ts[4] = {0, 0, 0, 0};
    bool ok;
    if (e->c.io_mem) {
        // Zero-copy: fill the bound tensor ourselves and run. No
        // rknn_inputs_set, so the driver's per-element widening never happens.
        if (preboxed_f16) {
            std::memcpy(e->c.io_mem->virt_addr, preboxed_f16,
                        e->input.size() * sizeof(uint16_t));
            ok = run_once_io_mem_filled(e->c, &e->hm, &e->off, &e->out, ts);
        } else {
            ok = run_once_io_mem(e->c, e->input.data(), e->input.size(), e->conv_div,
                                 &e->hm, &e->off, &e->out, ts);
        }
    } else {
        ok = run_once(e->c, e->input, &e->hm, &e->off, &e->out, ts,
                      preboxed_f16, preboxed_f16 ? e->input.size() : 0,
                      preboxed_f32, preboxed_f32 ? e->input.size() : 0,
                      f32_passthrough);
    }
    if (!ok) return false;
    if (t) {
        // ts = {inputs_set, rknn_run, outputs_get, dequant}
        t->inputs_set_ms = ts[0];
        t->run_ms = ts[1];
        t->outputs_get_ms = ts[2];
        t->rknn_ms = ts[0] + ts[1] + ts[2];
        t->dequant_ms = ts[3];
    }
    if (e->want_perf) {
        // Valid only after rknn_outputs_get, per the SDK header.
        //
        // run_duration is CUMULATIVE over every rknn_run since rknn_init -- it is
        // not the per-frame figure. Taking it raw gave 36.28 s for a 537-frame
        // run (36280.88/537 = 67.6 ms, i.e. the whole run). So take the delta
        // against the previous query; the first frame is discarded because it
        // absorbs everything accumulated during init.
        rknn_perf_run pr;
        std::memset(&pr, 0, sizeof(pr));
        if (rknn_query(e->c.ctx, RKNN_QUERY_PERF_RUN, &pr, sizeof(pr)) == 0) {
            const int64_t cur = pr.run_duration;
            if (e->last_run_us > 0 && cur > e->last_run_us)
                t->run_hw_ms = (double)(cur - e->last_run_us) / 1000.0;
            e->last_run_us = cur;
        }
    }
    // Release this frame's ~1.2 MB of tensor buffers before the next frame asks
    // RKNN for a fresh pair. Output-neutral (values come from rknn_run) and it
    // is what makes a long run safe.
    release(e->c, &e->out);

    const double t_pk0 = now_ms();
    extract_peaks(e->hm.data(), e->off.data(), e->c.hm_h, e->c.hm_w, e->stride, e->th, e->top_k,
                  &e->dets);
    if (t) t->peaks_ms = now_ms() - t_pk0;

    const char* base = out_name;
    e->detections += (long)e->dets.size();
    ++e->frames;

    const double t_w0 = now_ms();
    if (e->detf) {
        e->detf << base << "\t" << e->dets.size();
        for (size_t k = 0; k < e->dets.size(); k++)
            e->detf << "\t" << e->dets[k].x << " " << e->dets[k].y << " " << e->dets[k].score;
        e->detf << "\n";
        // Flush per frame so a consumer tailing the file sees progress.
        e->detf.flush();
    }

    if (!e->dump_dir.empty()) {
        std::string stem = e->dump_dir + "/" + e->tag + "_" + base;
        for (auto& ch : stem) if (ch == '.') ch = '_';
        if (stem.size() > 4) stem = stem.substr(0, stem.size() - 4);
        if (!write_bin(stem + "_hm.bin", e->hm.data(), e->hm.size() * 4) ||
            !write_bin(stem + "_off.bin", e->off.data(), e->off.size() * 4)) {
            std::fprintf(stderr, "[FATAL] heatmap dump write failed: %s\n", stem.c_str());
            return false;
        }
    }
    if (t) {
        t->write_ms = now_ms() - t_w0;
        t->total_ms = now_ms() - t_frame0;
        e->acc.load_ms += t->load_ms;
        e->acc.letterbox_ms += t->letterbox_ms;
        e->acc.rknn_ms += t->rknn_ms;
        e->acc.inputs_set_ms += t->inputs_set_ms;
        e->acc.run_ms += t->run_ms;
        e->acc.outputs_get_ms += t->outputs_get_ms;
        e->acc.dequant_ms += t->dequant_ms;
        e->acc.peaks_ms += t->peaks_ms;
        e->acc.write_ms += t->write_ms;
        e->acc.total_ms += t->total_ms;
        e->acc.run_hw_ms += t->run_hw_ms;
    }
    if (e->want_hash) {
        e->hash = fnv1a(e->hm.data(), e->hm.size() * sizeof(float)) ^ e->hash;
        e->hash = fnv1a(e->off.data(), e->off.size() * sizeof(float)) ^ e->hash;
    }
    if (n_det_out) *n_det_out = (long)e->dets.size();
    return true;
}

// File path: decode, then the shared body. `load_ms` is attributed here.
bool infer_one(InferEngine* e, const std::string& path, long* n_det_out, InferTiming* t) {
    int sw = 0, sh = 0, sc = 0;
    const double t_ld0 = now_ms();
    uint8_t* src = stbi_load(path.c_str(), &sw, &sh, &sc, 3);
    const double load_ms = now_ms() - t_ld0;
    if (!src) {
        std::fprintf(stderr, "[FATAL] stbi_load failed: %s\n", path.c_str());
        return false;
    }
    const char* slash = std::strrchr(path.c_str(), '/');
    const char* base = slash ? slash + 1 : path.c_str();
    const bool ok = infer_frame(e, src, sw, sh, nullptr, base, n_det_out, t);
    std::free(src);
    if (t && ok) t->load_ms = load_ms;
    if (ok) e->acc.load_ms += load_ms;
    return ok;
}

// No-I/O path: the caller already has the feature in stb-image RGB order.
bool infer_one_rgb(InferEngine* e, const uint8_t* rgb, int w, int h, long* n_det_out,
                   InferTiming* t) {
    char nm[64];
    std::snprintf(nm, sizeof(nm), "frame_%06ld", e->frames);
    return infer_frame(e, rgb, w, h, nullptr, nm, n_det_out, t);
}

bool letterbox_feature(const uint8_t* rgb, int w, int h, std::vector<uint8_t>* dst, double* ms) {
    const double t0 = now_ms();
    letterbox(rgb, w, h, kDefaultSize, dst);   // identical call, identical bytes
    if (ms) *ms = now_ms() - t0;
    return true;
}

bool infer_preboxed(InferEngine* e, const std::vector<uint8_t>& boxed, const char* name,
                    long* n_det_out, InferTiming* t) {
    return infer_frame(e, nullptr, 0, 0, boxed.data(), name, n_det_out, t);
}

// ---------------------------------------------------------------------------
// Pre-widened (fp16) input path.
//
// An fp16 graph fed RKNN_TENSOR_UINT8 makes the runtime widen all 640*640*3
// elements inside rknn_inputs_set -- measured 45.4 ms/frame, 29% of a 162 ms
// fp16 frame, and it sits ON the NPU worker's critical path. letterbox_feature_f16
// does the same widening on the letterbox thread instead, and this entry point
// uploads the result directly.
//
// The numbers handed over are the SAME 0..255 integers the uint8 path carries:
// the graph's std_values=[[255,255,255]] division lives inside the model, not in
// the upload, so the computed values must match. That is an expectation, not a
// guarantee -- callers must diff the detections TSV against the uint8 path to
// confirm it, and must not treat "fp16 is just a wider type" as the proof.
// ---------------------------------------------------------------------------
// float32 -> binary16, round-to-nearest-even. GCC 9.3 has no _Float16, and
// arm_neon.h's vcvt_f16_f32 needs FP16 support the toolchain headers here do
// not expose, so the conversion is done on the bits. Only non-negative inputs
// reach this in the letterbox path (uint8/255 in [0,1]), which is the easy
// direction; the negative branch is handled anyway so the helper is safe to
// reuse.
static inline uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) {          // Inf / NaN
        return (uint16_t)(sign | 0x7C00u | (man ? 0x200u : 0u));
    }
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);          // overflow -> Inf
    if (exp <= 0) {                                              // subnormal / zero
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);              // 24 - (exp + 10)
        uint32_t h = man >> shift;
        const uint32_t rem = man & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;         // round half to even
        return (uint16_t)(sign | h);
    }
    uint32_t h = (uint32_t)(exp << 10) | (man >> 13);
    const uint32_t rem = man & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;       // RNE
    return (uint16_t)(sign | h);
}

bool letterbox_feature_f16(const uint8_t* rgb, int w, int h, std::vector<uint16_t>* dst,
                           double* ms, bool div255) {
    const double t0 = now_ms();
    dst->resize((size_t)kDefaultSize * kDefaultSize * 3);
    if (div255) {
        // Model space: the runtime no longer applies std_values once we declare
        // FLOAT16, so the 0..255 integers have to be divided here.
        std::vector<uint8_t> boxed;
        letterbox(rgb, w, h, kDefaultSize, &boxed);
        for (size_t i = 0; i < boxed.size(); ++i)
            (*dst)[i] = f32_to_f16_bits((float)boxed[i] * (1.0f / 255.0f));
    } else {
        // Raw 0..255 integers, which is what this graph's input tensor wants
        // (scale = 1.0 -- the division lives inside the model). FUSED: the
        // canvas is written straight out as fp16, bit-identical to
        // letterbox() + u8->f16. (The previous version here wrote the raw
        // byte as if it were an fp16 bit pattern, which is simply wrong.)
        letterbox_to_f16(rgb, w, h, kDefaultSize, dst->data());
    }
    if (ms) *ms = now_ms() - t0;
    return true;
}

bool infer_preboxed_f16(InferEngine* e, const std::vector<uint16_t>& boxed_f16, const char* name,
                        long* n_det_out, InferTiming* t) {
    if (!e->c.input_is_f16) {
        std::fprintf(stderr, "[FATAL] infer_preboxed_f16 on a non-FLOAT16 model graph\n");
        return false;
    }
    // e->input still has to be filled: the all-zero guard below it reads that
    // buffer, and it is what the dump_input path serialises.
    e->input.resize(boxed_f16.size());
    for (size_t i = 0; i < boxed_f16.size(); ++i) e->input[i] = (uint8_t)boxed_f16[i];
    return infer_frame(e, nullptr, 0, 0, nullptr, name, n_det_out, t, boxed_f16.data());
}

// ---------------------------------------------------------------------------
// Pre-widened input path, FLOAT32 edition -- the one the driver actually accepts.
//
// Why this exists, measured end to end: an fp16 graph fed RKNN_TENSOR_UINT8 makes
// rknn_inputs_set widen and normalise all 640*640*3 elements on the CALLER'S
// thread -- 45.40 ms/frame measured on 34_1, 28% of a 162 ms fp16 frame, and it
// is pure CPU work sitting between two rknn_run calls on the NPU worker.
// pass_through=1 removes it, but the upload buffer must be exactly the model's
// own input size. The driver says so out loud when it is not:
//
//   E RKNN: rknn_inputs_set, param input size(2457600) < model input size(4915200)
//
// 4915200 = 640*640*3*4 bytes. So the runtime wants a float32 HOST buffer, even
// though RKNN_QUERY_INPUT_ATTR reports in_attr.type == FLOAT16 for this graph.
// A uint16 upload (2457600 B) can never be accepted -- it fails on every frame.
//
// `div255`: pass values already divided by 255 instead of raw 0..255 integers.
// With pass_through=1 the runtime applies NO conversion and therefore does NOT
// apply the graph's std_values=[[255,255,255]]. Which side of that division the
// model expects is NOT guessed here: both settings stay reachable, and the
// caller settles it by diffing the detections TSV against the uint8 path.
// ---------------------------------------------------------------------------
// uint8 -> float32, 16 elements per iteration. NEON is baseline on aarch64 and
// vcvtq_f32_u32 is plain NEON -- unlike the vcvt_f16_f32 that arm_neon.h could
// not offer above, which is why f32_to_f16_bits has to work on the bits.
//
// NOTE, deliberately not hidden: the div255 branch multiplies by 1/255f rather
// than dividing. That can differ from `v / 255.0f` by <=1 ulp on a few of the
// 256 possible byte values. The graph input is fp16, whose ulp is ~2^-10, so the
// fp16 rounding should absorb it -- but that is an argument, not a measurement,
// and the TSV diff is what actually settles it. If it ever does matter, this is
// the line to change.
static void u8_to_f32(const uint8_t* src, float* dst, size_t n, bool div255) {
    const float kScale = div255 ? (1.0f / 255.0f) : 1.0f;
    size_t i = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
    for (; i + 16 <= n; i += 16) {
        const uint8x16_t v = vld1q_u8(src + i);
        const uint16x8_t lo16 = vmovl_u8(vget_low_u8(v));
        const uint16x8_t hi16 = vmovl_u8(vget_high_u8(v));
        vst1q_f32(dst + i +  0, vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16))),  kScale));
        vst1q_f32(dst + i +  4, vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16))), kScale));
        vst1q_f32(dst + i +  8, vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16))),  kScale));
        vst1q_f32(dst + i + 12, vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16))), kScale));
    }
#endif
    for (; i < n; ++i) dst[i] = (float)src[i] * kScale;
}

bool letterbox_feature_f32(const uint8_t* rgb, int w, int h, F32Boxed* dst,
                           double* ms, bool div255) {
    const double t0 = now_ms();
    dst->u8.resize((size_t)kDefaultSize * kDefaultSize * 3);
    letterbox(rgb, w, h, kDefaultSize, &dst->u8);   // identical call, identical bytes
    dst->f32.resize(dst->u8.size());
    u8_to_f32(dst->u8.data(), dst->f32.data(), dst->u8.size(), div255);
    if (ms) *ms = now_ms() - t0;
    return true;
}

bool infer_preboxed_f16_io(InferEngine* e, const std::vector<uint16_t>& boxed_f16, const char* name,
                          long* n_det_out, InferTiming* t) {
    if (!e->c.io_mem) {
        std::fprintf(stderr, "[FATAL] infer_preboxed_f16_io without a bound io-mem\n");
        return false;
    }
    if (!e->c.input_is_f16) {
        std::fprintf(stderr, "[FATAL] infer_preboxed_f16_io on a non-FLOAT16 model graph\n");
        return false;
    }
    // e->input still has to be filled: the all-zero guard reads it.
    e->input.resize(boxed_f16.size());
    for (size_t i = 0; i < boxed_f16.size(); ++i) e->input[i] = (uint8_t)boxed_f16[i];
    return infer_frame(e, nullptr, 0, 0, nullptr, name, n_det_out, t, boxed_f16.data());
}

bool infer_preboxed_f32(InferEngine* e, const F32Boxed& b, const char* name,
                        long* n_det_out, InferTiming* t, bool passthrough) {
    if (!e->c.input_is_f16) {
        std::fprintf(stderr, "[FATAL] infer_preboxed_f32 on a non-FLOAT16 model graph\n");
        return false;
    }
    if (b.f32.size() != (size_t)kDefaultSize * kDefaultSize * 3) {
        std::fprintf(stderr, "[FATAL] infer_preboxed_f32: buffer is %zu elements, need %zu\n",
                     b.f32.size(), (size_t)kDefaultSize * kDefaultSize * 3);
        return false;
    }
    // The uint8 box travels in as `preboxed` so infer_frame's all-zero guard and
    // dump_input keep describing the real frame. Passing it the way the fp16 path
    // does would let a 0x0 letterbox clobber e->input and silently weaken that
    // check into a check of a grey placeholder.
    return infer_frame(e, nullptr, 0, 0, b.u8.data(), name, n_det_out, t, nullptr,
                       b.f32.data(), passthrough);
}

// Pin this context to specific NPU cores. RK3588 exposes three cores with mask
// values 1 / 2 / 4 (RKNN_NPU_CORE_0 / _1 / _2, combinable); the runtime's
// default is RKNN_NPU_CORE_AUTO.
//
// This exists because "two NPU workers" and "one worker per NPU core" are
// DIFFERENT claims, and only the second one is a measurement. Every earlier run
// used AUTO, so nothing so far established that worker 0 actually lands on core
// 0. Called right after infer_open and before the first rknn_run, which is the
// window the API allows.
//
// A non-zero return is reported, never swallowed: a core mask that silently
// failed to apply would make the whole "core 0 / core 1" statement hollow while
// the throughput numbers still looked fine.
int infer_set_npu_core(InferEngine* e, int mask) {
    if (!e || mask == 0) return 0;
    const int rc = rknn_set_core_mask(e->c.ctx, (rknn_core_mask)mask);
    if (rc != 0) {
        std::fprintf(stderr, "[FATAL] rknn_set_core_mask(0x%x) failed rc=%d\n", mask, rc);
        return rc;
    }
    e->c.core_mask = mask;
    std::printf("[2] NPU core mask = 0x%x\n", mask);
    return 0;
}

int infer_npu_core_mask(const InferEngine* e) { return e ? e->c.core_mask : 0; }

void infer_enable_hash(InferEngine* e) { e->want_hash = true; }
uint32_t infer_hash(const InferEngine* e) { return e->hash; }
void infer_stage_totals(const InferEngine* e, InferTiming* out) {
    if (!out) return;
    *out = e->acc;
    if (e->frames > 0) {
        const double n = (double)e->frames;
        out->load_ms /= n;       out->letterbox_ms /= n;
        out->rknn_ms /= n;       out->dequant_ms /= n;
        out->inputs_set_ms /= n; out->run_ms /= n;
        out->outputs_get_ms /= n;
        out->peaks_ms /= n;      out->write_ms /= n;
        out->total_ms /= n;
    }
}

long infer_frames_done(const InferEngine* e) { return e->frames; }
long infer_detections(const InferEngine* e) { return e->detections; }
void infer_close(InferEngine* e) {
    if (!e) return;
    if (e->detf) e->detf.close();
    rknn_destroy(e->c.ctx);
    delete e;
}

// =============================================================================
// Stage A -- feature extraction (batch = a loop over the stream above)
// =============================================================================
FeatureResult extract_features(const FeatureParams& p) {
    FeatureResult res;
    FeatureStream* s = feature_stream_open(p);
    if (!s) return res;
    std::string path;
    FrameTiming t;
    long n = 0;
    while (feature_stream_next(s, &path, &t)) {
        res.frames.push_back(path);
        if (++n % 100 == 0)
            std::printf("[FEAT] %s %ld/%ld\n", p.sequence.c_str(), n,
                        n + feature_stream_remaining(s));
    }
    res.md5 = feature_stream_md5(s);
    res.pushes = n;
    res.ok = true;
    long fg = 0, ch = 0, fd = 0, cc = 0;
    feature_stream_handles(s, &fg, &ch, &fd, &cc);
    std::printf("[SEQ] %-32s frames=%-6ld md5=%s | [HANDLE] fitgate=%ld chain=%ld "
                "fitdeadline=%ld cornercap=%ld\n", p.sequence.c_str(), n, res.md5.c_str(), fg, ch,
                fd, cc);
    feature_stream_close(s);
    return res;
}

// =============================================================================
// Stage B -- RKNN int8 inference
// =============================================================================
InferResult run_infer(const InferParams& p) {
    InferResult res;
    std::vector<std::string> frames = p.frames;
    if (frames.empty() && !p.batch_list.empty()) {
        FILE* f = std::fopen(p.batch_list.c_str(), "r");
        if (!f) {
            std::fprintf(stderr, "[FATAL] cannot read batch list %s\n", p.batch_list.c_str());
            return res;
        }
        char line[1024];
        while (std::fgets(line, sizeof(line), f)) {
            std::string s(line);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' '))
                s.pop_back();
            if (!s.empty()) frames.push_back(s);
        }
        std::fclose(f);
    }
    if (frames.empty()) {
        std::fprintf(stderr, "[FATAL] no frames to infer\n");
        return res;
    }

    // Deliberately a loop over infer_one() rather than a second copy of the
    // frame body: the serial demo and a pipelined demo then run THE SAME code
    // per frame, which is what makes "outputs identical" structural.
    InferEngine* e = infer_open(p);
    if (!e) return res;
    for (size_t i = 0; i < frames.size(); ++i) {
        if (!infer_one(e, frames[i])) {
            infer_close(e);
            return res;
        }
        if ((i + 1) % 100 == 0)
            std::printf("[INFER] %zu/%zu  dets=%ld\n", i + 1, frames.size(),
                        infer_detections(e));
    }
    res.frames = infer_frames_done(e);
    res.detections = infer_detections(e);
    res.ok = true;
    infer_close(e);
    std::printf("[DONE] frames=%ld detections=%ld\n", res.frames, res.detections);
    return res;
}

}  // namespace e2ecore
