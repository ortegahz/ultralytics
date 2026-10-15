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
// Stage A -- feature extraction
// =============================================================================
FeatureResult extract_features(const FeatureParams& p) {
    FeatureResult res;

    // Same self-test the binary runs first: a broken md5 must abort before any
    // feature is produced, never be mistaken for a feature mismatch later.
    std::string why;
    if (!gmcpp::md5_self_test(&why)) {
        std::fprintf(stderr, "[FATAL] built-in md5 self-test failed (%s)\n", why.c_str());
        return res;
    }

    std::string seq_dir;
    if (!find_sequence_folder(p.raw_root, p.sequence, &seq_dir)) {
        std::fprintf(stderr, "[FATAL] sequence folder not found for %s under %s\n",
                     p.sequence.c_str(), p.raw_root.c_str());
        return res;
    }
    std::vector<std::string> names = list_dir(seq_dir, true);
    std::sort(names.begin(), names.end(), natural_less);
    if (p.limit > 0 && static_cast<int>(names.size()) > p.limit)
        names.resize(static_cast<size_t>(p.limit));
    if (names.empty()) {
        std::fprintf(stderr, "[FATAL] no frames in %s\n", seq_dir.c_str());
        return res;
    }

    OnlineFeaturePipeline::Backend backend = OnlineFeaturePipeline::kCpu;
    if (p.fused == "cpu")       backend = OnlineFeaturePipeline::kCpu;
    else if (p.fused == "gpu")  backend = OnlineFeaturePipeline::kGpu;
    else if (p.fused == "both") backend = OnlineFeaturePipeline::kBoth;
    else {
        std::fprintf(stderr, "[FATAL] fused must be 'cpu', 'gpu' or 'both', got '%s'\n",
                     p.fused.c_str());
        return res;
    }

    std::unique_ptr<FusedGpu> gpu;
    if (backend != OnlineFeaturePipeline::kCpu) {
        std::printf("[BUILD] backend=%s anchor_step=%d opencv=%s\n", p.fused.c_str(),
                    p.anchor_step, CV_VERSION);
        // hw_linear / fp64_coord are the binary's defaults (0), i.e. false.
        gpu.reset(new FusedGpu(p.kernel_dir, false, false, p.window * p.stride_step + 3));
    }
    if (!p.out_dir.empty()) mkdir_p(p.out_dir);

    // Affinity stays at the binary's default ("none").
    const std::string aff = bind_cpus(parse_cpu_list("none"));
    std::fprintf(stderr, "[THREADS]   requested=%d affinity=%s (effective opencv threads=%d)\n",
                 p.threads, aff.c_str(), cv::getNumThreads());

    // ---- every remaining knob keeps the binary's frozen default -------------
    OnlineFeaturePipeline pipe(p.window, p.stride_step, p.anchor_step, p.downscale,
                               /*rng_seed_per_fit=*/false, p.mode == "nogmc", gpu.get(), backend,
                               /*pyr_cache=*/false, p.fit_corners, p.fit_win, p.fit_level,
                               p.fit_iters, p.cache_parts, p.threads);
    pipe.set_profiling(false);   // profiling is a timing ledger only; it cannot move pixels
    pipe.set_seq_id(0);
    if (p.gate_shift > 0.0) pipe.set_fit_gate(p.gate_shift, p.gate_min, p.gate_max);
    pipe.gpu_prealloc(0);
    pipe.gpu_pin_pad(false);
    pipe.set_corner_probe(p.corner_probe != 0);
    pipe.set_fit_deadline(0.0, false);   // deterministic mode: output independent of load

    std::printf("[FIT-GATE] shift<=%.2f x frame  scale in [%.2f,%.2f]\n", p.gate_shift,
                p.gate_min, p.gate_max);
    std::printf("ring_depth=%ld lags=%d..%d\n", pipe.ring_depth(), pipe.lags().front(),
                pipe.lags().back());
    std::printf("[BUILD] window=%d stride_step=%d anchor_step=%d downscale=%d mode=%s opencv=%s\n",
                p.window, p.stride_step, p.anchor_step, p.downscale, p.mode.c_str(), CV_VERSION);

    gmcpp::MD5 seq_md5;
    res.frames.reserve(names.size());

    for (size_t fi = 0; fi < names.size(); ++fi) {
        const std::string path = seq_dir + "/" + names[fi];
        pipe.set_seq_frame(static_cast<int>(fi));
        const Mat frame = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (frame.empty()) {
            std::fprintf(stderr, "[FATAL] failed to read %s\n", path.c_str());
            return res;
        }

        const Mat out = pipe.push(frame, nullptr);

        // Fingerprint the PRE-ENCODE planes: comparing decoded BMPs would only
        // measure the encoder, not feature drift. Same rule as the binary.
        gmcpp::MD5 f;
        f.update(reinterpret_cast<const uint8_t*>(out.data),
                 static_cast<size_t>(out.total()));
        seq_md5.update(f.hex());

        // Builder layout: HxWx3 stacked in [Ch0, Ch1, Ch2] order.
        Mat hwc(frame.rows, frame.cols, CV_8UC3);
        for (int y = 0; y < frame.rows; ++y)
            for (int x = 0; x < frame.cols; ++x) {
                const int q = y * frame.cols + x;
                hwc.at<cv::Vec3b>(y, x)[0] = out.at<uint8_t>(0, q);
                hwc.at<cv::Vec3b>(y, x)[1] = out.at<uint8_t>(1, q);
                hwc.at<cv::Vec3b>(y, x)[2] = out.at<uint8_t>(2, q);
            }
        char nm[256];
        std::snprintf(nm, sizeof(nm), "%s_%06zu.%s", p.sequence.c_str(), fi, p.out_ext.c_str());
        const std::string full = p.out_dir + "/" + nm;
        if (!cv::imwrite(full, hwc)) {
            std::fprintf(stderr, "[FATAL] failed to write %s\n", full.c_str());
            return res;
        }
        res.frames.push_back(full);
        if ((fi + 1) % 100 == 0)
            std::printf("[FEAT] %s %zu/%zu\n", p.sequence.c_str(), fi + 1, names.size());
    }

    const auto& s = pipe.stats();
    const double n = static_cast<double>(std::max<long>(1, s.n_push));
    std::printf("[SEQ] %-32s frames=%-6zu md5=%s  fits/frm=%.2f warps/frm=%.2f compose/frm=%.2f\n",
                p.sequence.c_str(), names.size(), seq_md5.hex().c_str(),
                static_cast<double>(s.n_fit) / n, static_cast<double>(s.n_warp) / n,
                static_cast<double>(s.n_compose) / n);
    if (s.n_fit_rejected || pipe.n_chain_rejected() || pipe.n_fit_deadline())
        std::printf("[HANDLE] fitgate=%ld chain=%ld fitdeadline=%ld cornercap=%ldf/%ldcalls\n",
                    s.n_fit_rejected, pipe.n_chain_rejected(), pipe.n_fit_deadline(),
                    pipe.corner_cap_frames(), pipe.corner_capped());

    res.md5 = seq_md5.hex();
    res.pushes = static_cast<long>(names.size());
    res.ok = true;
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
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) s.pop_back();
            if (!s.empty()) frames.push_back(s);
        }
        std::fclose(f);
    }
    if (frames.empty()) {
        std::fprintf(stderr, "[FATAL] no frames to infer\n");
        return res;
    }

    std::vector<uint8_t> model;
    if (!read_file(p.model_path.c_str(), &model)) {
        std::fprintf(stderr, "[FATAL] cannot read model %s\n", p.model_path.c_str());
        return res;
    }
    std::printf("[1] model: %zu bytes, fnv1a32=%08x\n", model.size(), fnv1a(model.data(), model.size()));

    Ctx c;
    if (rknn_init(&c.ctx, model.data(), (uint32_t)model.size(), 0u, NULL) != 0) {
        std::fprintf(stderr, "[FATAL] rknn_init failed\n");
        return res;
    }
    std::printf("[2] rknn_init ok\n");

    rknn_input_output_num io;
    std::memset(&io, 0, sizeof(io));
    rknn_query(c.ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
    if (io.n_input != 1 || io.n_output != 2) {
        std::fprintf(stderr, "[FATAL] expected 1 input / 2 outputs (heatmap, offset)\n");
        rknn_destroy(c.ctx);
        return res;
    }

    rknn_tensor_attr in_attr;
    std::memset(&in_attr, 0, sizeof(in_attr));
    in_attr.index = 0;
    rknn_query(c.ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
    if (in_attr.fmt == RKNN_TENSOR_NCHW) {
        c.in_c = (int)in_attr.dims[1]; c.in_h = (int)in_attr.dims[2]; c.in_w = (int)in_attr.dims[3];
    } else {
        c.in_h = (int)in_attr.dims[1]; c.in_w = (int)in_attr.dims[2]; c.in_c = (int)in_attr.dims[3];
    }
    if (c.in_h != kDefaultSize || c.in_w != kDefaultSize || c.in_c != 3) {
        std::fprintf(stderr, "[FATAL] model input is %dx%dx%d, expected 640x640x3\n",
                     c.in_h, c.in_w, c.in_c);
        rknn_destroy(c.ctx);
        return res;
    }

    c.n_out = io.n_output;
    c.out_attr.assign(c.n_out, rknn_tensor_attr{});
    for (uint32_t i = 0; i < c.n_out; i++) {
        std::memset(&c.out_attr[i], 0, sizeof(rknn_tensor_attr));
        c.out_attr[i].index = i;
        rknn_query(c.ctx, RKNN_QUERY_OUTPUT_ATTR, &c.out_attr[i], sizeof(c.out_attr[i]));
    }
    {
        const rknn_tensor_attr& a = c.out_attr[0];
        if (a.n_dims != 4 || a.dims[0] != 1 || a.dims[1] != 1) {
            std::fprintf(stderr, "[FATAL] out[0] is not [1,1,H,W]\n");
            rknn_destroy(c.ctx);
            return res;
        }
        c.hm_h = (int)a.dims[2]; c.hm_w = (int)a.dims[3];
        const rknn_tensor_attr& b = c.out_attr[1];
        if (b.n_dims != 4 || b.dims[0] != 1 || b.dims[1] != 2) {
            std::fprintf(stderr, "[FATAL] out[1] is not [1,2,H,W]\n");
            rknn_destroy(c.ctx);
            return res;
        }
        c.off_h = (int)b.dims[2]; c.off_w = (int)b.dims[3];
        if (c.off_h != c.hm_h || c.off_w != c.hm_w) {
            std::fprintf(stderr, "[FATAL] heatmap %dx%d and offset %dx%d grids disagree\n",
                         c.hm_h, c.hm_w, c.off_h, c.off_w);
            rknn_destroy(c.ctx);
            return res;
        }
        if (c.in_h / c.hm_h != p.stride || c.in_w / c.hm_w != p.stride) {
            std::fprintf(stderr, "[FATAL] input/grid ratio %d does not match --stride %d\n",
                         c.in_h / c.hm_h, p.stride);
            rknn_destroy(c.ctx);
            return res;
        }
        std::printf("    heatmap %dx%d, offset %dx%d, implied stride %d\n", c.hm_h, c.hm_w,
                    c.off_h, c.off_w, c.in_h / c.hm_h);
    }
    std::printf("[3] batch mode: %zu frames, th=%.4f stride=%d topk=%d\n", frames.size(), p.th,
                p.stride, p.top_k);

    if (!p.dump_dir.empty()) mkdir_p(p.dump_dir.c_str());

    std::ofstream detf;
    if (!p.detections_out.empty()) {
        detf.open(p.detections_out, std::ios::out | std::ios::trunc);
        if (!detf) {
            std::fprintf(stderr, "[FATAL] cannot open --detections-out %s\n", p.detections_out.c_str());
            rknn_destroy(c.ctx);
            return res;
        }
    }

    std::vector<uint8_t> input;
    std::vector<float> hm, off;
    std::vector<rknn_output> out;
    std::vector<Det> dets;

    for (size_t fi = 0; fi < frames.size(); ++fi) {
        const std::string& path = frames[fi];
        int sw = 0, sh = 0, sc = 0;
        uint8_t* src = stbi_load(path.c_str(), &sw, &sh, &sc, 3);
        if (!src) {
            std::fprintf(stderr, "[FATAL] stbi_load failed: %s\n", path.c_str());
            rknn_destroy(c.ctx);
            return res;
        }
        letterbox(src, sw, sh, kDefaultSize, &input);
        std::free(src);

        // Guard on the MAX, not the min: infrared frames are full of zero
        // background, so `mn == 0` would reject every legitimate image.
        int mx = 0;
        for (uint8_t v : input) mx = std::max(mx, (int)v);
        if (mx == 0) {
            std::fprintf(stderr, "[FATAL] letterboxed input is all zero: %s\n", path.c_str());
            rknn_destroy(c.ctx);
            return res;
        }

        if (p.dump_input) {
            const char* slash = std::strrchr(path.c_str(), '/');
            std::string stem_name = slash ? slash + 1 : path;
            for (auto& ch : stem_name) if (ch == '.') ch = '_';
            char path_buf[1200];
            std::snprintf(path_buf, sizeof(path_buf), "%s%s%s.input.bin", p.input_dump_dir.c_str(),
                          p.input_dump_dir.empty() || p.input_dump_dir.back() == '/' ? "" : "/",
                          stem_name.c_str());
            if (mkdir(p.input_dump_dir.c_str(), 0755) != 0 && errno != EEXIST) { /* exists */ }
            if (!write_bin(path_buf, input.data(), input.size())) {
                std::fprintf(stderr, "[FATAL] could not write input dump %s\n", path_buf);
                rknn_destroy(c.ctx);
                return res;
            }
        }

        if (!run_once(c, input, &hm, &off, &out, nullptr)) {
            rknn_destroy(c.ctx);
            return res;
        }
        // Free this frame's 1.2 MB of tensor buffers before the next frame asks
        // RKNN for a fresh pair. The binary does not do this in batch mode and
        // therefore leaks one set per frame; releasing here is output-neutral
        // (the values are produced by rknn_run, not by the release) and is what
        // makes this stage safe to call from a long-running parallel demo.
        release(c, &out);

        extract_peaks(hm.data(), off.data(), c.hm_h, c.hm_w, p.stride, p.th, p.top_k, &dets);

        const char* base = std::strrchr(path.c_str(), '/');
        base = base ? base + 1 : path.c_str();
        res.detections += (long)dets.size();

        if (detf) {
            detf << base << "\t" << dets.size();
            for (size_t k = 0; k < dets.size(); k++)
                detf << "\t" << dets[k].x << " " << dets[k].y << " " << dets[k].score;
            detf << "\n";
        }

        if (!p.dump_dir.empty()) {
            std::string stem = p.dump_dir + "/" + p.tag + "_" + base;
            for (auto& ch : stem) if (ch == '.') ch = '_';
            if (stem.size() > 4) stem = stem.substr(0, stem.size() - 4);   // strip "_bmp"
            if (!write_bin(stem + "_hm.bin", hm.data(), hm.size() * 4) ||
                !write_bin(stem + "_off.bin", off.data(), off.size() * 4)) {
                std::fprintf(stderr, "[FATAL] heatmap dump write failed: %s\n", stem.c_str());
                rknn_destroy(c.ctx);
                return res;
            }
        }

        res.frames = (long)fi + 1;
        if ((fi + 1) % 100 == 0)
            std::printf("[INFER] %zu/%zu  dets=%ld\n", fi + 1, frames.size(), res.detections);
    }

    if (detf) detf.close();
    rknn_destroy(c.ctx);
    std::printf("[DONE] frames=%ld detections=%ld\n", res.frames, res.detections);
    res.ok = true;
    return res;
}

}  // namespace e2ecore
