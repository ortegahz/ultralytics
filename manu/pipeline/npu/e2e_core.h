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
    // Enables RKNN_FLAG_COLLECT_PERF_MASK + RKNN_QUERY_PERF_RUN so rknn_ms can
    // be split into "NPU compute" vs "tensor upload/readback". Inflates the run,
    // so it is a DIAGNOSTIC mode and must not be used for headline FPS.
    bool rknn_perf = false;
};

struct InferResult {
    long frames = 0;
    long detections = 0;   // total detections across all frames
    bool ok = false;
};

// Runs the NPU stage. Same letterbox / run_once / dequant / extract_peaks as
// rknn_heatmap_smoke.cpp. Returns 0 on success.
InferResult run_infer(const InferParams& p);

// -----------------------------------------------------------------------------
// Streaming form -- what a pipelined demo needs
//
// The batch functions above are fine for "run the whole sequence", but a demo
// that overlaps stage A with stage B has to interleave at FRAME granularity:
// while the NPU chews on frame N, the feature pipeline must already be
// producing frame N+1. So the stages are exposed as pull/push objects whose
// per-frame bodies are the same code the batch path runs.
//
// Both are opaque: they hold classes that live in an anonymous namespace inside
// the production sources, which a header cannot name.
// -----------------------------------------------------------------------------

struct FeatureStream;   // opaque
struct InferEngine;     // opaque

struct FrameTiming {
    double total_ms = 0;   // fit + warp + median (+ gpu tail), per frame
    double fit_ms = 0;
    double warp_ms = 0;
    double median_ms = 0;
};

// Per-frame NPU-side cost, split the same way rknn_heatmap_smoke.cpp's [7]
// block splits it, so a pipelined run can be compared against a serial one
// without either side being re-measured differently.
struct InferTiming {
    double load_ms = 0;    // stbi_load (file read + BMP decode)
    double letterbox_ms = 0;
    double rknn_ms = 0;    // inputs_set + rknn_run + outputs_get (the sum of the
                           // three below; kept so older reports stay comparable)
    // rknn_ms split three ways, straight off run_once()'s own t_stage[]:
    // how much of the NPU call is tensor upload, hardware run, tensor readback.
    double inputs_set_ms = 0;    // rknn_inputs_set: host -> NPU-DRAM copy
    double run_ms = 0;           // rknn_run: the actual NPU execution
    double outputs_get_ms = 0;   // rknn_outputs_get: NPU-DRAM -> host copy
    double dequant_ms = 0;
    double peaks_ms = 0;   // extract_peaks
    double write_ms = 0;    // det TSV + heatmap dumps
    double total_ms = 0;   // wall clock for the frame inside infer_one
    // RKNN's own "real inference time" in ms, from RKNN_QUERY_PERF_RUN. It is a
    // different quantity from rknn_ms above: rknn_ms is the wall clock around
    // inputs_set + rknn_run + outputs_get, so the gap between them is the
    // upload/readback cost, and this number is what the NPU spent computing.
    // Only filled when InferParams::rknn_perf is set (it needs
    // RKNN_FLAG_COLLECT_PERF_MASK, which inflates rknn_run -- diagnostic only).
    double run_hw_ms = 0;
};

// Opens the feature pipeline on one sequence. Returns nullptr on failure.
FeatureStream* feature_stream_open(const FeatureParams& p);

// Produces ONE frame: runs the GMC fit/warp/median, writes the three-channel
// BMP, and hands back its path. Returns false when the sequence is exhausted.
// `timing` may be null.
//
// The BMP write is deliberately kept. cv::imwrite stores BGR and stb_image
// reads RGB, so the disk round trip is what defines the byte order the NPU
// stage sees. Handing the in-memory plane straight to run_infer instead would
// silently swap the channels and change every heatmap.
bool feature_stream_next(FeatureStream* s, std::string* out_bmp, FrameTiming* timing);

// Same frame, but handed over as an in-memory RGB buffer instead of a BMP path,
// so a speed run can skip the encode+decode entirely.
//
// The byte order is NOT the pipeline's natural [Ch0,Ch1,Ch2]. It is what
// stbi_load would have returned for the BMP that cv::imwrite wrote: OpenCV
// stores BGR, stb_image returns RGB, so the model sees [Ch2,Ch1,Ch0]. Passing
// the planes in their natural order instead would silently swap two channels
// and change every heatmap -- which is precisely why infer_enable_hash exists.
bool feature_stream_next_rgb(FeatureStream* s, std::vector<uint8_t>* rgb, int* w, int* h,
                             FrameTiming* timing);

// Rolling md5 over the pre-encode channel planes, and how many frames remain.
std::string feature_stream_md5(FeatureStream* s);
long feature_stream_remaining(FeatureStream* s);

// Cumulative handle census, so the pipelined run can assert it degraded no
// further than the serial one (bad-case handling must stay ON).
void feature_stream_handles(FeatureStream* s, long* fitgate, long* chain,
                            long* fit_deadline, long* corner_cap);
void feature_stream_close(FeatureStream* s);

// Opens the NPU once (rknn_init) and keeps the context for the whole run.
InferEngine* infer_open(const InferParams& p);

// Runs ONE feature through the model. Returns false on failure.
// `t` may be null; when given it receives the per-frame cost breakdown.
bool infer_one(InferEngine* e, const std::string& feature_bmp, long* n_det_out = nullptr,
               InferTiming* t = nullptr);

// -------------------------------------------------------------------------
// No-I/O path -- what the pipeline can actually do when nobody asks it to
// write intermediate files.
//
// The file path is not free: cv::imwrite encodes a 512x512x3 BMP and
// stbi_load decodes it again, every frame. That round trip is ALSO what fixes
// the channel order the NPU sees: OpenCV stores BGR, stb_image returns RGB, so
// the model is fed [Ch2, Ch1, Ch0] of what the feature pipeline produced.
// Callers bypassing the disk must therefore pass the SAME byte order, which is
// what infer_one_rgb expects -- and which hash-verify proves is equivalent.
//
// infer_one_rgb takes the feature already in that RGB order and runs the
// identical letterbox -> run_once -> dequant -> extract_peaks chain.
// -------------------------------------------------------------------------
bool infer_one_rgb(InferEngine* e, const uint8_t* rgb, int w, int h,
                   long* n_det_out = nullptr, InferTiming* t = nullptr);

// -------------------------------------------------------------------------
// Split-letterbox stage.
//
// Profiling says the NPU thread spends 28.60 ms/frame in letterbox (a float
// bilinear 512x512 -> 640x640, pure CPU) with the hardware idle the whole time,
// and only 59.96 ms in rknn. So the "NPU stage" is 94 ms but the accelerator
// is busy 60 of them. Lifting letterbox onto its own pipeline stage lets the
// accelerator run back to back.
//
// It produces byte-identical output by construction: it calls the SAME
// letterbox() with the SAME arguments. Nothing about the pixels changes; only
// which thread does the arithmetic.
// -------------------------------------------------------------------------
bool letterbox_feature(const uint8_t* rgb, int w, int h, std::vector<uint8_t>* dst,
                       double* ms = nullptr);

// Runs a frame whose letterbox was already done by letterbox_feature().
bool infer_preboxed(InferEngine* e, const std::vector<uint8_t>& boxed, const char* name,
                    long* n_det_out = nullptr, InferTiming* t = nullptr);

// Rolling fnv1a over every frame's (heatmap, offset) tensors. This is how the
// no-I/O path is PROVEN equivalent to the file path without writing a byte:
// both modes print the same 32-bit chain hash iff every tensor matched.
void infer_enable_hash(InferEngine* e);
uint32_t infer_hash(const InferEngine* e);

// Cumulative NPU sub-stage totals, for attributing where a frame goes.
void infer_stage_totals(const InferEngine* e, InferTiming* out);

long infer_frames_done(const InferEngine* e);
long infer_detections(const InferEngine* e);
void infer_close(InferEngine* e);

}  // namespace e2ecore

#endif  // E2E_CORE_H
