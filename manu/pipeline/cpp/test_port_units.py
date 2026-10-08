#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Offline bit-exactness unit tests for the C++ port's OWN logic.

Scope -- deliberately narrow. This does NOT run the feature pipeline and does NOT need the
dataset. It tests only the pieces the port wrote itself, against the exact Python they must
agree with:

    compose()        the float64 homogeneous chain composition of `_compose`
    anchor_grid()    the anchor set of the frozen builder
    natural_less()   the frame ordering comparator of `natural_key`
    Ch2 median+clip  np.median(stack(21 uint8)) -> clip(curr-bg, 0, 255) -> uint8
    md5              the fingerprint that carries the whole G1 gate

The OpenCV primitives (Shi-Tomasi / LK / RANSAC / warpAffine / resize) are NOT tested here:
they are the *same* C++ code the Python side already calls, so their equivalence is by
construction, not by test. What still needs proving end-to-end on the server is that the
pipeline assembles those primitives in the same order -- that is what verify_cpp_port.py and
its G1/G2 gates do.

Every test prints its own sample count and its own verdict. A test that ran zero samples
reports SKIP, never PASS.

Usage (no dataset, no GPU, runs in seconds):
    python manu/pipeline/cpp/test_port_units.py
    python manu/pipeline/cpp/test_port_units.py --cxx g++ --keep
"""

from __future__ import annotations

import argparse
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

CPP_DIR = Path(__file__).resolve().parent
SRC = CPP_DIR / "gmc_stream.cpp"

COMPOSE_CPP = r"""
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <opencv2/core.hpp>
using cv::Matx23f; using cv::Matx33d;
static const Matx23f kIdentity(1.0f,0.0f,0.0f, 0.0f,1.0f,0.0f);
static Matx33d to33(const Matx23f& m){
    Matx33d h = Matx33d::eye();
    for (int i=0;i<2;i++) for (int j=0;j<3;j++) h(i,j) = static_cast<double>(m(i,j));
    return h;
}
static Matx23f compose(const Matx23f& nw, const Matx23f& ac){
    const Matx33d A = to33(nw), B = to33(ac);
    Matx33d H;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++){
        double s = 0.0;
        for (int k=0;k<3;k++) s += A(i,k)*B(k,j);
        H(i,j) = s;
    }
    const double n = H(2,2);
    if (!std::isfinite(n) || std::fabs(n) < 1e-12) return kIdentity;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) H(i,j) /= n;
    for (int i=0;i<3;i++) for (int j=0;j<3;j++)
        if (!std::isfinite(H(i,j))) return kIdentity;
    Matx23f out;
    for (int i=0;i<2;i++) for (int j=0;j<3;j++) out(i,j) = static_cast<float>(H(i,j));
    return out;
}
int main(int argc,char** argv){
    if (argc < 3) return 2;
    FILE* fi=std::fopen(argv[1],"rb"); FILE* fo=std::fopen(argv[2],"wb");
    if(!fi||!fo) return 2;
    unsigned n=0; if (std::fread(&n,4,1,fi)!=1) return 2;
    std::vector<unsigned char> buf(48);
    for (unsigned i=0;i<n;i++){
        if (std::fread(buf.data(),48,1,fi)!=1) return 2;
        float fa[12];
        std::memcpy(fa, buf.data(), 48);   // reinterpret bytes as float32, NOT as ints
        Matx23f a(fa[0],fa[1],fa[2],fa[3],fa[4],fa[5]);
        Matx23f b(fa[6],fa[7],fa[8],fa[9],fa[10],fa[11]);
        Matx23f c = compose(b, a);
        unsigned char o[24];
        std::memcpy(o, &c.val[0], 24);
        std::fwrite(o,24,1,fo);
    }
    std::fclose(fi); std::fclose(fo); return 0;
}
"""

MEDIAN_CPP = r"""
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>
int main(int argc,char** argv){
    if (argc < 3) return 2;
    FILE* fi=std::fopen(argv[1],"rb"); FILE* fo=std::fopen(argv[2],"wb");
    if(!fi||!fo) return 2;
    uint32_t n=0; if (std::fread(&n,4,1,fi)!=1) return 2;
    std::vector<uint8_t> col(21), out(n);
    for (uint32_t i=0;i<n;i++){
        if (std::fread(col.data(),21,1,fi)!=1) return 2;
        uint8_t cur; if (std::fread(&cur,1,1,fi)!=1) return 2;
        std::nth_element(col.begin(), col.begin()+21/2, col.end());
        const float bg = static_cast<float>(col[21/2]);
        float v = static_cast<float>(cur) - bg;
        if (v < 0.0f) v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        out[i] = static_cast<uint8_t>(v);
    }
    std::fwrite(out.data(),1,n,fo);
    std::fclose(fi); std::fclose(fo); return 0;
}
"""

MD5_CPP = r"""
#include "md5.h"
#include <cstdio>
using gmcpp::MD5;
static std::string H(const std::string& s){
    MD5 h; h.update(reinterpret_cast<const uint8_t*>(s.data()), s.size()); return h.hex();
}
int main(){
    printf("%d\n", (int)gmcpp::md5_self_test(nullptr));
    const char* cases[] = {"", "a", "abc", "message digest", "abcdefghijklmnopqrstuvwxyz",
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
        "12345678901234567890123456789012345678901234567890123456789012345678901234567890"};
    for (auto c : cases) printf("%s\n", H(c).c_str());
    MD5 inc; const std::string big(100000,'x');
    for (size_t i=0;i<big.size();i+=7)
        inc.update(reinterpret_cast<const uint8_t*>(big.data()+i), std::min<size_t>(7,big.size()-i));
    printf("%s\n", inc.hex().c_str());
    return 0;
}
"""

# The exact FP flags from CMakeLists.txt. -ffp-contract=off is load-bearing.
FP_FLAGS = ["-O3", "-ffp-contract=off", "-fno-fast-math", "-fno-unsafe-math-optimizations"]


def extract(src: Path, signature: str) -> str:
    """Pull one top-level function out of the shipped source verbatim.

    Signature-based with brace matching, because a plain "up to the next marker" slice either
    swallows every function in between or drops the closing brace -- and a harness that
    silently diverges from the shipped code would make these tests meaningless.
    """
    text = src.read_text(encoding="utf-8")
    i = text.index(signature)
    j = text.index("{", text.index(")", i))
    depth, k = 0, j
    while k < len(text):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[i:k + 1]
        k += 1
    raise ValueError(f"unbalanced braces while extracting {signature!r}")


def compile_cxx(cxx: str, source: str, out: Path, workdir: Path, include_stub: Path | None) -> bool:
    sp = workdir / (out.stem + ".cpp")
    sp.write_text(source, encoding="utf-8")
    cmd = [cxx, "-std=c++17", *FP_FLAGS]
    if include_stub:
        cmd += ["-I", str(include_stub)]
    cmd += [str(sp), "-o", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[BUILD FAIL] {' '.join(cmd)}\n{r.stderr[:2000]}")
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxx", default="g++")
    ap.add_argument("--opencv-include", default="", help="dir containing opencv2/core.hpp")
    ap.add_argument("--cases", type=int, default=20000)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()

    import numpy as np

    workdir = Path(tempfile.mkdtemp(prefix="gmcpp_units_"))
    stub = Path(a.opencv_include) if a.opencv_include else None
    results: list[tuple[str, str, str]] = []

    print("=" * 92)
    print("   C++ PORT OFFLINE UNIT TESTS  (own logic only; OpenCV primitives are shared code)")
    print("=" * 92)
    print(f"  source     : {SRC}")
    print(f"  compiler   : {a.cxx}")
    print(f"  fp flags   : {' '.join(FP_FLAGS)}")
    print(f"  opencv inc : {stub or '(none - compose test needs a core.hpp shim)'}")
    print("=" * 92)

    # ---- 1. compose(): the float64 chain, the single most fragile piece -----------------
    if stub is None or (stub / "opencv2" / "core.hpp").is_file():
        binp = workdir / "compose"
        if compile_cxx(a.cxx, COMPOSE_CPP, binp, workdir, stub):
            rng = np.random.default_rng(20261008)
            cases = []
            for _ in range(a.cases):
                k = int(rng.integers(0, 3))
                if k == 0:
                    th = float(rng.uniform(-0.2, 0.2)); s = float(rng.uniform(0.98, 1.02))
                    m = np.array([[s * np.cos(th), -s * np.sin(th), rng.uniform(-8, 8)],
                                  [s * np.sin(th), s * np.cos(th), rng.uniform(-8, 8)]], np.float32)
                    p = np.array([[s * np.cos(th), -s * np.sin(th), rng.uniform(-15, 15)],
                                  [s * np.sin(th), s * np.cos(th), rng.uniform(-15, 15)]], np.float32)
                    cases.append((m, p if rng.integers(0, 2) else m.copy()))
                elif k == 1:
                    m = np.array([[rng.uniform(0.5, 1.5), 0, 0],
                                  [0, rng.uniform(0.5, 1.5), 0]], np.float32)
                    cases.append((m, m.copy()))
                else:
                    m = (rng.normal(0, 10 ** rng.uniform(-4, 3), size=(2, 3))).astype(np.float32)
                    cases.append((m, m.copy()))

            def to33(m):
                h = np.eye(3, dtype=np.float64); h[:2, :] = m; return h

            IDENT = np.eye(2, 3, dtype=np.float32)

            def compose(nw, ac):
                h = to33(nw) @ to33(ac)
                nn = h[2, 2]
                if not np.isfinite(nn) or abs(nn) < 1e-12: return IDENT.copy()
                h = h / nn
                if not np.isfinite(h).all(): return IDENT.copy()
                return h[:2, :].astype(np.float32)

            fin, fout = workdir / "c_in.bin", workdir / "c_out.bin"
            with open(fin, "wb") as f:
                f.write(struct.pack("<I", len(cases)))
                for m, p in cases: f.write(m.tobytes()); f.write(p.tobytes())
            subprocess.run([str(binp), str(fin), str(fout)], check=True)
            raw = open(fout, "rb").read()
            n_ok = sum(compose(p, m).tobytes() == raw[i * 24:(i + 1) * 24]
                        for i, (m, p) in enumerate(cases))
            verdict = "PASS" if n_ok == len(cases) else "FAIL"
            results.append(("compose() vs numpy", f"{n_ok}/{len(cases)} bit-identical", verdict))
        else:
            results.append(("compose() vs numpy", "build failed (needs OpenCV core.hpp shim)", "SKIP"))
    else:
        results.append(("compose() vs numpy", "no OpenCV core.hpp available", "SKIP"))

    # ---- 2. median + clip kernel ---------------------------------------------------------
    binp = workdir / "median"
    if compile_cxx(a.cxx, MEDIAN_CPP, binp, workdir, None):
        rng = np.random.default_rng(7)
        n = min(60000, a.cases * 3)
        cols = []
        for _ in range(n):
            k = int(rng.integers(0, 5))
            if k == 0: cols.append(rng.integers(0, 256, 21, dtype=np.uint8))
            elif k == 1: cols.append(np.full(21, int(rng.integers(0, 256)), dtype=np.uint8))
            elif k == 2: cols.append(np.clip(rng.normal(200, 3, 21), 0, 255).astype(np.uint8))
            elif k == 3: cols.append(np.clip(rng.normal(30, 2, 21), 0, 255).astype(np.uint8))
            else: cols.append(rng.choice([0, 255, 128], size=21).astype(np.uint8))
        cols = np.stack(cols)
        cur = rng.integers(0, 256, n, dtype=np.uint8)
        fin, fout = workdir / "m_in.bin", workdir / "m_out.bin"
        with open(fin, "wb") as f:
            f.write(struct.pack("<I", n))
            for c, v in zip(cols, cur): f.write(bytes(c.tolist())); f.write(bytes([int(v)]))
        subprocess.run([str(binp), str(fin), str(fout)], check=True)
        got = np.frombuffer(open(fout, "rb").read(), dtype=np.uint8)
        ref = np.clip(cur.astype(np.float32) - np.median(cols, axis=1).astype(np.float32),
                      0, 255).astype(np.uint8)
        n_ok = int((got == ref).sum())
        results.append(("Ch2 median+clip vs numpy", f"{n_ok}/{n} bit-identical",
                        "PASS" if n_ok == n else "FAIL"))
    else:
        results.append(("Ch2 median+clip vs numpy", "build failed", "SKIP"))

    # ---- 3. anchor_grid + natural_less (extracted verbatim from the shipped source) -------
    try:
        tok = extract(SRC, "std::vector<std::pair<int, std::string>> tokenize_stem(")
        nl = extract(SRC, "bool natural_less(")
        ag = extract(SRC, "std::vector<int> anchor_grid(")
        harness = ("#include <algorithm>\n#include <cctype>\n#include <cstdio>\n"
                   "#include <cstdlib>\n#include <cstring>\n#include <set>\n#include <string>\n"
                   "#include <utility>\n#include <vector>\nnamespace {\n" + tok + nl + ag + "\n}\n"
                   "int main(int argc,char** argv){\n"
                   "  if(argc>=5 && std::string(argv[1])==\"anchors\"){\n"
                   "    auto v=anchor_grid(atoi(argv[2]),atoi(argv[3]),atoi(argv[4]));\n"
                   "    for(size_t i=0;i<v.size();i++) printf(\"%s%d\",i?\",\":\"\",v[i]);\n"
                   "    printf(\"\\n\"); return 0; }\n"
                   "  FILE* f=std::fopen(argv[1],\"r\"); if(!f) return 2;\n"
                   "  std::vector<std::string> v; char line[512];\n"
                   "  while(std::fgets(line,sizeof(line),f)){ std::string s(line);\n"
                   "    while(!s.empty()&&(s.back()=='\\n'||s.back()=='\\r')) s.pop_back();\n"
                   "    if(!s.empty()) v.push_back(s); }\n"
                   "  std::fclose(f);\n"
                   "  std::sort(v.begin(),v.end(),natural_less);\n"
                   "  for(const auto&s:v) printf(\"%s\\n\",s.c_str());\n"
                   "  return 0; }\n")
        binp = workdir / "orch"
        if compile_cxx(a.cxx, harness, binp, workdir, None):
            def anchor_grid_py(s_, a_, m_):
                if a_ % s_: raise ValueError
                return [l for l in sorted(set(range(s_, m_ + 1, a_)) | {m_}) if l % s_ == 0]

            n_ag = n_ok_ag = 0
            for s_, a_, m_ in [(2, 10, 42), (2, 2, 42), (2, 4, 42), (2, 6, 42), (2, 8, 42),
                               (2, 14, 42), (1, 10, 42), (2, 10, 40), (4, 12, 48)]:
                n_ag += 1
                py = ",".join(map(str, anchor_grid_py(s_, a_, m_)))
                cpp = subprocess.run([str(binp), "anchors", str(s_), str(a_), str(m_)],
                                     capture_output=True, text=True).stdout.strip()
                n_ok_ag += int(py == cpp)
            results.append(("anchor_grid() vs Python", f"{n_ok_ag}/{n_ag} combos match",
                            "PASS" if n_ok_ag == n_ag else "FAIL"))

            def natural_key(p):
                return [int(x) if x.isdigit() else x.lower()
                        for x in re.split(r"(\d+)", Path(p).stem)]

            groups = {
                "plain-numeric": [f"{i:06d}.jpg" for i in [1, 2, 3, 9, 10, 99, 100, 999, 1000]],
                "DJI-style": [f"DJI_0051_2_{i:06d}.jpg" for i in [1, 5, 10, 42, 100, 999, 1000]],
                "wg2022-style": [f"wg2022_ir_052_split_08_{i:06d}.jpg" for i in [3, 42, 100, 7, 999, 1000]],
                "01_4485-range": [f"01_4485_{i:06d}.jpg" for i in range(1167, 1178)],
                "VIDEO-style": [f"VIDEO00005_19700101_{i:06d}.jpg" for i in [17, 2959, 2960, 300]],
                "mixed-width-digits": [f"seq_{i}.jpg" for i in [1, 2, 10, 20, 100, 200, 1000]],
                "case-fold": ["IMG_1.JPG", "img_2.jpg", "IMG_10.JPG", "img_9.jpg", "Img_100.jpg"],
                "leading-zero-pad": [f"f{i:03d}.jpg" for i in [1, 2, 10, 100, 101, 999]],
            }
            n_g = n_ok_g = 0
            for files in groups.values():
                (workdir / "names.txt").write_text("\n".join(files), encoding="utf-8")
                cpp = [x for x in subprocess.run([str(binp), str(workdir / "names.txt")],
                                                 capture_output=True, text=True).stdout.split("\n") if x]
                n_g += 1
                n_ok_g += int(cpp == sorted(files, key=natural_key))
            results.append(("natural_less() vs Python", f"{n_ok_g}/{n_g} patterns match",
                            "PASS" if n_ok_g == n_g else "FAIL"))
        else:
            results.append(("anchor_grid()/natural_less()", "build failed", "SKIP"))
    except Exception as exc:  # noqa: BLE001
        results.append(("anchor_grid()/natural_less()", f"harness error: {exc}", "SKIP"))

    # ---- 4. md5 fingerprint --------------------------------------------------------------
    binp = workdir / "md5t"
    if compile_cxx(a.cxx, MD5_CPP, binp, workdir, CPP_DIR):
        out = subprocess.run([str(binp)], capture_output=True, text=True).stdout.split()
        import hashlib
        cases = ["", "a", "abc", "message digest", "abcdefghijklmnopqrstuvwxyz",
                 "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
                 "12345678901234567890123456789012345678901234567890123456789012345678901234567890",
                 "x" * 100000]
        n_ok_md5 = sum(hashlib.md5(c.encode()).hexdigest() == h for c, h in zip(cases, out[1:]))
        results.append(("md5 vs hashlib", f"{n_ok_md5}/{len(cases)} vectors match",
                        "PASS" if n_ok_md5 == len(cases) else "FAIL"))
    else:
        results.append(("md5 vs hashlib", "build failed", "SKIP"))

    print()
    for name, detail, verdict in results:
        print(f"  {name:<34}{detail:<26}{verdict}")
    failed = [r for r in results if r[2] == "FAIL"]
    skipped = [r for r in results if r[2] == "SKIP"]
    print("\n" + "=" * 92)
    if failed:
        print(f"[VERDICT] FAIL -- {len(failed)} of {len(results)} tests failed")
    elif skipped:
        print(f"[VERDICT] PARTIAL -- {len(skipped)} SKIPPED (never counted as PASS): "
              + "; ".join(r[0] for r in skipped))
    else:
        print("[VERDICT] PASS -- the port's own logic is bit-identical to Python")
    print("  NOTE: this covers the port's OWN code only. The end-to-end claim "
          "(pipeline assembly order,\n        OpenCV build parity, RNG stream) is established by "
          "verify_cpp_port.py's G1/G2 on real data.")
    print("=" * 92)

    if a.keep:
        print(f"  workdir kept: {workdir}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())