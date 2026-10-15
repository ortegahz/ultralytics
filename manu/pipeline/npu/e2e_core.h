// =============================================================================
// e2e_core.h -- reusable stages of the board end-to-end demo
//
// WHY THIS EXISTS
// ---------------
// run_hm_demo.sh orchestrates two CERTIFIED BINARIES over SSH plus a host-side
// python rename step. That is fine for measurement, but it makes the pipeline
// impossible to reuse: nothing can call "GMC stage" or "NPU stage" as a function,
// so a second demo (e.g. one that overlaps the two stages, or one that shards
// frames across workers) has to re-implement both, and any re-implementation
// risks drifting from the numbers the binaries produced.
//
// This header exposes exactly those two stages as functions. They are NOT
// rewrites: e2e_core.cpp #includes the two production .cpp files with their
// main() compiled out and calls the same classes / same functions
// (OnlineFeaturePipeline::push, letterbox, run_once, dequant, extract_peaks).
// Identical output is therefore structural, not something to be re-verified
// every time.
//
// A future parallel demo can do:
//     auto f1 = e2ecore::extract_features(p);          // stage A
//     e2ecore::run_infer({..., frames = shard_a});    // stage B, per worker
// =============================================================================
#ifndef E2E_CORE_H
#define E2E_CORE_H

#include <string>
#include <vector>

namespace e2ecore {

// -----------------------------------------------------------------------------
// Stage A -- GMC feature extraction: raw infrared sequence -> 3-channel features
// -----------------------------------------------------------------------------
struct FeatureParams {
    std::string raw_root;       // directory holding <sequence>/  (board side)
    std::string sequence;       // sequence name
    std::string out_dir;        // where the feature BMPs are written
    std::string out_ext = "bmp";// "bmp" on the board (its OpenCV has no png/jpg writer)
    std::string kernel_dir = "/root/kernels";
    std::string fused = "gpu";  // cpu | gpu | both  (the shipping arm is "gpu")
    std::string mode = "tree";  // tree | nogmc

    // GMC fit knobs. These defaults are the FROZEN values: omitting every knob
    // reproduces the shipped baseline byte for byte.
    int window = 21;
    int stride_step = 2;
    int anchor_step = 10;
    int downscale = 2;
    int fit_corners = 600;
    int fit_win = 15;
    int fit_level = 2;
    int fit_iters = 20;
    int cache_parts = 7;        // resize | corners | pyramid
    int threads = 1;
    double gate_shift = 0.25;   // max centre shift, in frame widths
    double gate_min = 0.5;      // fit scale bounds
    double gate_max = 2.0;
    int corner_probe = 1;
    int limit = 0;              // 0 = all frames
    bool timing = false;
};

struct FeatureResult {
    std::vector<std::string> frames;  // written feature BMPs, in push order
    std::string md5;                  // rolling md5 over the PRE-ENCODE channel planes
    long pushes = 0;
    bool ok = false;
};

// Runs the feature pipeline over one sequence and returns the feature paths in
// push order, so the caller can feed them straight into run_infer without any
// rename/symlink step. Returns 0 on success.
FeatureResult extract_features(const FeatureParams& p);

// -----------------------------------------------------------------------------
// Stage B -- RKNN int8 inference: 3-channel features -> heatmap -> detections
// -----------------------------------------------------------------------------
struct InferParams {
    std::string model_path;
    std::vector<std::string> frames;  // feature BMP paths; empty => read batch_list
    std::string batch_list;           // alternative to frames (one path per line)
    float th = 0.25f;
    int stride = 2;
    int top_k = 100;
    std::string dump_dir;             // writes <tag>_<stem>_hm.bin / _off.bin
    std::string detections_out;       // TSV: <stem>\t<ndet>\t<x y score>...
    std::string tag = "frame";
    bool dump_input = false;          // also write the exact uint8 NHWC NPU input
    std::string input_dump_dir;
};

struct InferResult {
    long frames = 0;
    long detections = 0;   // total detections across all frames
    bool ok = false;
};

// Runs the NPU stage. Same letterbox / run_once / dequant / extract_peaks as
// rknn_heatmap_smoke.cpp. Returns 0 on success.
InferResult run_infer(const InferParams& p);

}  // namespace e2ecore

#endif  // E2E_CORE_H
