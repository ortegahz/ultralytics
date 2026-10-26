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
    // Zero-copy input: bind the model's input tensor to host memory with
    // rknn_set_io_mem and never call rknn_inputs_set. On an fp16 graph the
    // runtime otherwise widens all 640*640*3 elements on the calling thread
    // (measured 33.6 ms/frame on the board, 11.1 ms pinned to an A76).
    bool use_io_mem = false;
    // Value written into the input tensor = host byte / conv_div. Measured on
    // the board: the fp16 graph's input tensor has scale=1.0, so it holds the
    // RAW 0..255 integers and conv_div must stay 1.0 (255 gives zero
    // detections -- the division happens inside the graph).
    float conv_div = 1.0f;
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

// -------------------------------------------------------------------------
// Pre-widened input path (fp16 graphs only).
//
// Feeding an fp16 graph as UINT8 makes rknn_inputs_set widen all 640*640*3
// elements itself -- measured 45.4 ms/frame, 29% of a 162 ms fp16 frame, and it
// sits on the NPU worker's critical path. These two move that widening onto the
// letterbox thread.
//
// The values handed over are the same 0..255 integers the uint8 path carries
// (the graph's std_values division is inside the model, not the upload). That is
// an expectation the caller must CONFIRM by diffing the detections TSV against
// the uint8 path -- it is not established by the type change alone.
// infer_preboxed_f16 refuses to run on a non-FLOAT16 graph.
// -------------------------------------------------------------------------
// `div255`: pass 0..1 instead of raw 0..255.
//
// MEASURED, and the choice is not cosmetic: handing the runtime raw 0..255
// while declaring RKNN_TENSOR_FLOAT16 produced **zero detections** on all 537
// frames (process exit 0 -- a silent-wrong-output trap), which says the FLOAT16
// declaration makes the runtime STOP applying the graph's std_values=[[255]].
// So the caller has to hand over values already in model space. This flag
// exists so the semantics can be settled by measurement instead of by reading
// the SDK header, and so a future re-test costs one build instead of two.
// FUSED letterbox -> fp16: writes the padded canvas straight out as fp16,
// bit-identical to letterbox() followed by the u8->f16 conversion (the
// no-resize path is vectorised; the resize path keeps letterbox()'s exact
// rounding and converts afterwards). div255 selects the value space: false =
// raw 0..255 integers (what the board's fp16 graph wants, scale=1.0), true =
// already divided by 255.
bool letterbox_feature_f16(const uint8_t* rgb, int w, int h, std::vector<uint16_t>* dst,
                           double* ms = nullptr, bool div255 = false);
bool infer_preboxed_f16(InferEngine* e, const std::vector<uint16_t>& boxed_f16, const char* name,
                        long* n_det_out = nullptr, InferTiming* t = nullptr);

// Zero-copy variant: the pre-widened fp16 box is copied straight into the
// bound input tensor and the NPU is run without rknn_inputs_set. The copy is
// 2.4 MB of plain memcpy on the worker's own thread.
bool infer_preboxed_f16_io(InferEngine* e, const std::vector<uint16_t>& boxed_f16, const char* name,
                          long* n_det_out = nullptr, InferTiming* t = nullptr);

// -------------------------------------------------------------------------
// Pre-widened input path, FLOAT32 edition -- the one the runtime accepts.
//
// An fp16 graph fed RKNN_TENSOR_UINT8 makes rknn_inputs_set widen and normalise
// all 640*640*3 elements on the caller's thread: measured 45.40 ms/frame on
// 34_1, 28% of a 162 ms fp16 frame, pure CPU work between two rknn_run calls.
// pass_through=1 removes it -- provided the upload is exactly the model's own
// input size, which for this graph is 640*640*3*4 = 4915200 BYTES of float32,
// even though RKNN_QUERY_INPUT_ATTR reports in_attr.type == FLOAT16:
//
//   E RKNN: rknn_inputs_set, param input size(2457600) < model input size(4915200)
//
// A uint16 upload is rejected on every frame, so the uint16 variant above is a
// documented dead end, not an alternative.
//
// `div255`: hand over values already divided by 255. pass_through=1 performs no
// conversion, so the graph's std_values=[[255,255,255]] is NOT applied for us.
// Which side the model wants is settled by diffing the detections TSV against
// the uint8 path -- not by assumption, and not by the type declaration.
//
// `u8` is kept alongside `f32` so the all-zero guard and dump_input still see
// the real frame's bytes.
//
// `passthrough` selects the upload mode, and the default is the one that is
// MEASURED to be broken: pass_through=1 passes the size check and then SIGSEGVs
// inside librknnrt's memcpy. Passing false uploads the same buffer as an
// ordinary RKNN_TENSOR_FLOAT32 input instead.
// -------------------------------------------------------------------------
struct F32Boxed {
    std::vector<uint8_t> u8;    // byte-identical to letterbox_feature()'s output
    std::vector<float>   f32;   // the upload buffer
};

bool letterbox_feature_f32(const uint8_t* rgb, int w, int h, F32Boxed* dst,
                           double* ms = nullptr, bool div255 = false);
bool infer_preboxed_f32(InferEngine* e, const F32Boxed& b, const char* name,
                        long* n_det_out = nullptr, InferTiming* t = nullptr,
                        bool passthrough = true);

// Pin a context to specific NPU cores, e.g. one worker per core on RK3588.
// Mask values are rknn_core_mask bits: CORE_0 = 1, CORE_1 = 2, CORE_2 = 4,
// combinable; 0 leaves the runtime's default (RKNN_NPU_CORE_AUTO) alone.
// Must be called after infer_open and before the first rknn_run. Returns 0 on
// success and logs FATAL + returns non-zero if the runtime refuses -- the
// distinction matters, because an unapplied mask leaves the throughput numbers
// valid but the "worker N owns core N" claim unproven.
int infer_set_npu_core(InferEngine* e, int mask);
int infer_npu_core_mask(const InferEngine* e);

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
