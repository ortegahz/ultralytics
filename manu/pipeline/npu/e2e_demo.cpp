// =============================================================================
// e2e_demo.cpp -- single-process end-to-end demo on the RK3588 board.
//
// Replaces run_hm_demo.sh. No shell, no SSH, no host-side python rename step:
// one binary takes raw infrared in and writes three-channel features plus the
// inferred heatmap out.
//
//   raw infrared  --[ e2ecore::extract_features ]-->  3-channel feature BMP
//                --[ e2ecore::run_infer         ]-->  heatmap/offset + detections
//
// Both stages come from e2e_core.cpp, which #includes the two production
// sources with main() compiled out. The output is therefore byte-identical to
// the two-binary shell demo BY CONSTRUCTION -- not "close", identical -- because
// it is the same code, not a re-derivation of it.
//
// Usage on the board:
//   ./e2e_demo --raw-root raw --sequence 34_1 \
//              --out-root /mnt/manu/npu/out/demo/34_1 \
//              --model /mnt/manu/npu/data/trial0474_hm_nop0_int8.rknn \
//              --th 0.25 --kernel-dir /root/kernels
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "e2e_core.h"

namespace {

void usage(const char* argv0) {
    std::printf(
        "Usage: %s --raw-root DIR --sequence NAME --out-root DIR [options]\n"
        "\n"
        "  --raw-root   DIR   directory holding <sequence>/ raw infrared frames\n"
        "  --sequence   NAME  sequence to run\n"
        "  --out-root   DIR   output root (feat/, hm/, npu_det.tsv are created here)\n"
        "  --model      PATH  .rknn heatmap detector\n"
        "  --kernel-dir DIR   OpenCL kernel pair (default /root/kernels)\n"
        "  --fused      W     cpu | gpu | both          (default gpu)\n"
        "  --th         F     peak threshold            (default 0.25)\n"
        "  --stride     N     output stride             (default 2)\n"
        "  --topk       N     max detections per frame  (default 100)\n"
        "  --anchor-step N    GMC anchor step           (default 10 = the shipping arm)\n"
        "  --limit      N     cap frames, 0 = all      (default 0)\n"
        "  --no-heatmap      skip heatmap .bin dump (only features + detections)\n"
        "  --dump-input      also dump the exact uint8 NHWC NPU input\n",
        argv0);
}

const char* need(int argc, char** argv, int& i, const char* name) {
    if (i + 1 >= argc) {
        std::fprintf(stderr, "[FATAL] %s needs a value\n", name);
        std::exit(2);
    }
    return argv[++i];
}

}  // namespace

int main(int argc, char** argv) {
    std::string raw_root, sequence, out_root, model, kernel_dir = "/root/kernels";
    std::string fused = "gpu", out_ext = "bmp";
    float th = 0.25f;
    int stride = 2, top_k = 100, anchor_step = 10, limit = 0, dump_heatmap = 1, dump_input = 0;

    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (!std::strcmp(a, "--raw-root"))        raw_root = need(argc, argv, i, "--raw-root");
        else if (!std::strcmp(a, "--sequence"))   sequence = need(argc, argv, i, "--sequence");
        else if (!std::strcmp(a, "--out-root"))   out_root = need(argc, argv, i, "--out-root");
        else if (!std::strcmp(a, "--model"))      model = need(argc, argv, i, "--model");
        else if (!std::strcmp(a, "--kernel-dir")) kernel_dir = need(argc, argv, i, "--kernel-dir");
        else if (!std::strcmp(a, "--fused"))      fused = need(argc, argv, i, "--fused");
        else if (!std::strcmp(a, "--th"))         th = (float)std::atof(need(argc, argv, i, "--th"));
        else if (!std::strcmp(a, "--stride"))     stride = std::atoi(need(argc, argv, i, "--stride"));
        else if (!std::strcmp(a, "--topk"))       top_k = std::atoi(need(argc, argv, i, "--topk"));
        else if (!std::strcmp(a, "--anchor-step")) anchor_step = std::atoi(need(argc, argv, i, "--anchor-step"));
        else if (!std::strcmp(a, "--limit"))      limit = std::atoi(need(argc, argv, i, "--limit"));
        else if (!std::strcmp(a, "--no-heatmap")) dump_heatmap = 0;
        else if (!std::strcmp(a, "--dump-input")) dump_input = 1;
        else if (!std::strcmp(a, "-h") || !std::strcmp(a, "--help")) { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "[FATAL] unknown arg: %s\n", a); usage(argv[0]); return 2; }
    }

    if (raw_root.empty() || sequence.empty() || out_root.empty() || model.empty()) {
        std::fprintf(stderr, "[FATAL] --raw-root, --sequence, --out-root and --model are required\n");
        usage(argv[0]);
        return 2;
    }

    std::printf("=====================================================================\n");
    std::printf(" RK3588 e2e demo (single process, pure C++)\n");
    std::printf(" sequence : %s\n", sequence.c_str());
    std::printf(" raw-root : %s\n", raw_root.c_str());
    std::printf(" out-root : %s\n", out_root.c_str());
    std::printf(" model    : %s\n", model.c_str());
    std::printf(" th=%.4f stride=%d topk=%d anchor_step=%d fused=%s\n", th, stride, top_k,
                anchor_step, fused.c_str());
    std::printf("=====================================================================\n");

    // ---- Stage A: raw infrared -> three-channel features --------------------
    e2ecore::FeatureParams fp;
    fp.raw_root = raw_root;
    fp.sequence = sequence;
    fp.out_dir = out_root + "/feat";
    fp.out_ext = out_ext;
    fp.kernel_dir = kernel_dir;
    fp.fused = fused;
    fp.anchor_step = anchor_step;
    fp.limit = limit;

    const e2ecore::FeatureResult fr = e2ecore::extract_features(fp);
    if (!fr.ok) {
        std::fprintf(stderr, "[FATAL] stage A (feature extraction) failed\n");
        return 1;
    }
    std::printf("\n[A] features: %ld frames, md5=%s\n\n", fr.pushes, fr.md5.c_str());

    // ---- Stage B: features -> heatmap -> detections -------------------------
    // The feature paths come straight out of stage A, so no rename/symlink step
    // is needed: the shell demo needed one only because two separate processes
    // had to agree on a file name.
    e2ecore::InferParams ip;
    ip.model_path = model;
    ip.frames = fr.frames;
    ip.th = th;
    ip.stride = stride;
    ip.top_k = top_k;
    ip.detections_out = out_root + "/npu_det.tsv";
    ip.tag = sequence;
    if (dump_heatmap) {
        ip.dump_dir = out_root + "/hm";
        ip.input_dump_dir = out_root + "/npu_input";
        ip.dump_input = dump_input;
    }

    const e2ecore::InferResult ir = e2ecore::run_infer(ip);
    if (!ir.ok) {
        std::fprintf(stderr, "[FATAL] stage B (NPU inference) failed\n");
        return 1;
    }

    std::printf("\n=====================================================================\n");
    std::printf(" E2E_DONE  sequence=%s frames=%ld feature_md5=%s detections=%ld\n",
                sequence.c_str(), ir.frames, fr.md5.c_str(), ir.detections);
    std::printf("   features     : %s/feat\n", out_root.c_str());
    std::printf("   heatmap dump : %s/hm  (*_hm.bin / *_off.bin)\n", out_root.c_str());
    std::printf("   detections   : %s/npu_det.tsv\n", out_root.c_str());
    std::printf("=====================================================================\n");
    return 0;
}
