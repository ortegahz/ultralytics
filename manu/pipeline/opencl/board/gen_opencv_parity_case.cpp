// ===========================================================================
// gen_opencv_parity_case.cpp -- x86 only: turn real frames into raw grayscale
// for opencv_parity to consume.
//
// Split from opencv_parity.cpp on purpose. Decoding JPEG needs imgcodecs, and
// imgcodecs cannot be cross-compiled for this board -- the sysroot has no
// libjpeg/libpng/zlib headers at all. Keeping the decoder on the x86 side means
// the arm64 build needs only the six modules that were actually compiled, and
// both machines are guaranteed to read byte-identical pixels from the same
// file rather than each decoding independently.
//
// Colour handling matches the pipeline: the working image is grayscale.
// ===========================================================================

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

int main(int argc, char** argv) {
    std::string seq_dir, out_dir;
    int frames = 24;          // 24 frames = 23 fits, > kWindow so median engages
    int width = 640, height = 512;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "[ERROR] %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--seq-dir") seq_dir = next("--seq-dir");
        else if (a == "--out-dir") out_dir = next("--out-dir");
        else if (a == "--frames") frames = std::atoi(next("--frames").c_str());
    }
    if (seq_dir.empty() || out_dir.empty()) {
        std::fprintf(stderr,
                     "usage: %s --seq-dir DIR --out-dir DIR [--frames N]\n"
                     "  seq-dir holds 000001.jpg, 000002.jpg, ...\n", argv[0]);
        return 2;
    }

    std::printf("=== gen_opencv_parity_case ===\n");
    std::printf("OpenCV version : %s\n", CV_VERSION);
    std::printf("source         : %s\n", seq_dir.c_str());
    std::printf("destination    : %s\n", out_dir.c_str());
    std::printf("frames wanted  : %d\n\n", frames);

    // Same reasoning as opencv_parity: the destination lives on NFS and may
    // not exist yet. Creating it here keeps the two programs symmetric.
    ::mkdir(out_dir.c_str(), 0777);

    int written = 0;
    for (int i = 1; i <= frames; ++i) {
        char jp[1024], op[1024];
        std::snprintf(jp, sizeof jp, "%s/%06d.jpg", seq_dir.c_str(), i);
        std::snprintf(op, sizeof op, "%s/frame_%04d.gray", out_dir.c_str(), i - 1);

        const cv::Mat bgr = cv::imread(jp, cv::IMREAD_COLOR);
        if (bgr.empty()) {
            std::fprintf(stderr, "[ERROR] cannot read %s\n", jp);
            return 2;
        }
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        // Guard against a sequence whose geometry drifts mid-run: the parity
        // comparison assumes one fixed shape across all frames.
        if (i == 1) { width = gray.cols; height = gray.rows; }
        else if (gray.cols != width || gray.rows != height) {
            std::fprintf(stderr,
                         "[ERROR] frame %d is %dx%d, expected %dx%d\n",
                         i, gray.cols, gray.rows, width, height);
            return 2;
        }

        FILE* f = std::fopen(op, "wb");
        if (!f) { std::fprintf(stderr, "[ERROR] cannot write %s\n", op); return 2; }
        std::fwrite(gray.data, 1, gray.total(), f);
        std::fclose(f);
        std::printf("  wrote %s  (%dx%d)\n", op, width, height);
        ++written;
    }

    std::printf("\n[ok] %d raw frames of %dx%d 8UC1\n", written, width, height);
    return 0;
}
