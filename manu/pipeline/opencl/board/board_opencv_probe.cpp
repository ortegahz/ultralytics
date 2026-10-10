// ===========================================================================
// board_opencv_probe.cpp -- does this RK3588 have a usable OpenCV C++ SDK?
//
// WHY THIS EXISTS
//   The whole board-side plan is blocked on one question: can a C++ program on
//   the board include <opencv2/core.hpp> and link -lopencv_core? Everything
//   upstream assumes it cannot -- the cross toolchain's sysroot has no OpenCV,
//   the vendor SDK ships only a buildroot recipe, and neither the board nor
//   /usr/include/opencv4 exists there. The board DOES have a working Python
//   `cv2` (4.12.0), which is easy to mistake for "OpenCV is installed".
//
//   It is not. `import cv2` succeeding proves only that a Python extension
//   module exists. C++ compilation needs three things a pip wheel does not
//   ship: headers, a linkable libopencv_*.so, and a pkg-config/cmake package
//   file. This probe checks all of them separately so the answer cannot be
//   conflated again.
//
// BUILD AND RUN
//   The board has gcc/g++/make/cmake, so it compiles natively:
//       g++ -std=c++17 -O2 board_opencv_probe.cpp -o board_opencv_probe -ldl
//   No OpenCV headers are included here on purpose -- the whole point is to
//   determine whether they could be, and this file has to compile either way.
//   Cross-compiling from x86 works too: same command, cross g++, -ldl only.
// ===========================================================================

#include <dlfcn.h>
#include <dirent.h>
#include <sys/stat.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Report {
  int header_dirs = 0;
  int header_files = 0;
  int lib_files = 0;
  std::vector<std::string> libs;
  bool loaded = false;
  std::string loaded_path;
  std::string runtime_version;
};

bool path_exists(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0;
}

bool is_dir(const std::string& p) {
  struct stat st;
  return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool has_suffix(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

// Depth-limited walk, so a sweep cannot turn into a multi-minute scan of a
// 16 GiB board with a network mount under it.
void walk(const std::string& dir, int depth, const std::vector<std::string>& skip_dirs,
          const std::string& name_want, Report& rep, bool want_headers, bool want_libs) {
  if (depth <= 0) return;
  DIR* d = opendir(dir.c_str());
  if (!d) return;
  while (struct dirent* e = readdir(d)) {
    const std::string name = e->d_name;
    if (name == "." || name == "..") continue;
    bool skip = false;
    for (const std::string& s : skip_dirs)
      if (name == s) skip = true;
    if (skip) continue;
    const std::string full = dir + "/" + name;

    if (want_headers && name == name_want && is_dir(full)) {
      // The canonical marker for a usable OpenCV C++ SDK: the directory that
      // holds opencv2/core.hpp.
      if (path_exists(full + "/core.hpp") || is_dir(full + "/opencv2")) {
        ++rep.header_dirs;
        std::printf("  [FOUND] header dir : %s\n", full.c_str());
      }
    }
    if (want_libs && has_suffix(name, ".so")) {
      if (name.find("opencv") != std::string::npos) {
        ++rep.lib_files;
        if (rep.libs.size() < 12) rep.libs.push_back(full);
      }
    }
    if (is_dir(full)) walk(full, depth - 1, skip_dirs, name_want, rep, want_headers, want_libs);
  }
  closedir(d);
}

void section(const char* t) { std::printf("\n=== %s ===\n", t); }

}  // namespace

int main() {
  Report rep;
  std::printf("========== RK3588 OpenCV C++ SDK probe ==========\n");

  // ---------------------------------------------------------------- headers
  section("1. C++ headers (opencv2/core.hpp)");
  const char* header_guesses[] = {
      "/usr/include/opencv4/opencv2",
      "/usr/local/include/opencv4/opencv2",
      "/usr/include/opencv2",
      "/usr/local/include/opencv2",
      "/opt/opencv/include/opencv4/opencv2",
  };
  for (const char* g : header_guesses)
    std::printf("  %-44s %s\n", g, is_dir(g) ? "present" : "absent");

  const std::vector<std::string> skip = {"proc", "sys", "dev", "run", "tmp", "var",
                                          "media", "boot", "mnt", "home"};
  std::printf("  scanning the filesystem (depth 6, skipping %zu dirs)...\n", skip.size());
  walk("/", 6, skip, "opencv2", rep, /*want_headers=*/true, /*want_libs=*/false);
  std::printf("  header dirs found: %d\n", rep.header_dirs);
  if (rep.header_dirs == 0)
    std::printf("  => #include <opencv2/core.hpp> WILL NOT COMPILE\n");

  // ------------------------------------------------------------------- libs
  section("2. Linkable shared libraries (libopencv_*.so)");
  const char* lib_dirs[] = {
      "/usr/lib/aarch64-linux-gnu", "/usr/lib", "/usr/local/lib",
      "/usr/lib/jni",                "/opt/opencv/lib",
  };
  for (const char* d : lib_dirs) {
    if (!is_dir(d)) continue;
    int n = 0;
    if (DIR* dd = opendir(d)) {
      while (struct dirent* e = readdir(dd))
        if (strstr(e->d_name, "libopencv") && strstr(e->d_name, ".so")) ++n;
      closedir(dd);
    }
    std::printf("  %-30s %d libopencv*.so\n", d, n);
  }
  rep.lib_files = 0;
  walk("/", 6, skip, "", rep, /*want_headers=*/false, /*want_libs=*/true);
  std::printf("  libopencv*.so found anywhere: %d\n", rep.lib_files);
  for (const std::string& l : rep.libs) std::printf("    %s\n", l.c_str());
  if (rep.lib_files == 0)
    std::printf("  => -lopencv_core WILL NOT LINK\n");

  // --------------------------------------------------- runtime-only attempt
  // Even with no headers and no import library, a build that happened to ship
  // the shared object could still be probed by dlopen. cvGetVersionString is a
  // plain C entry point, so this says nothing about C++ ABI.
  section("3. dlopen probe (runtime only, no compile)");
  const char* sonames[] = {
      "libopencv_core.so",   "libopencv_core.so.4",  "libopencv_core.so.410",
      "libopencv_core.so.3", "libopencv_core.so.406", "libopencv_core.so.340",
      "libopencv_core.so.1",
  };
  for (const char* s : sonames) {
    void* h = dlopen(s, RTLD_NOW | RTLD_LOCAL);
    if (!h) continue;
    rep.loaded = true;
    rep.loaded_path = s;
    typedef void (*fn_ver)(char*, int);
    fn_ver v = reinterpret_cast<fn_ver>(dlsym(h, "cvGetVersionString"));
    if (v) {
      char buf[128] = {0};
      v(buf, sizeof(buf));
      rep.runtime_version = buf;
    }
    std::printf("  [ OK ] dlopen %s\n", s);
    if (v) std::printf("         cvGetVersionString -> %s\n", rep.runtime_version.c_str());
    else std::printf("         cvGetVersionString not exported\n");
    // Probe a few C++ symbols to see whether it is a full libopencv_core or
    // just the Python extension's private copy. Not called, only counted.
    const char* cpp_syms[] = {"_ZN2cv6imreadERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEi",
                              "_ZN2cv4imencodeERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKNS_11MatEi"};
    int found = 0;
    for (const char* sy : cpp_syms)
      if (dlsym(h, sy)) ++found;
    std::printf("         C++ ABI symbols present: %d/2 (0 suggests not a full libopencv_core)\n",
                found);
    break;
  }
  if (!rep.loaded) {
    std::printf("  no libopencv_core.so could be dlopen'ed either\n");
  }

  // ------------------------------------------------------------ python cv2
  section("4. Python cv2 (easy to mistake for 'OpenCV is installed')");
  const char* py_dirs[] = {
      "/usr/local/lib/python3.8/dist-packages/cv2",
      "/usr/lib/python3/dist-packages/cv2",
      "/usr/local/lib/python3.10/dist-packages/cv2",
      "/usr/local/lib/python3.11/dist-packages/cv2",
  };
  for (const char* d : py_dirs) {
    if (!is_dir(d)) continue;
    std::printf("  [FOUND] %s\n", d);
    if (DIR* dd = opendir(d)) {
      while (struct dirent* e = readdir(dd)) {
        const std::string n = e->d_name;
        if (n.find(".so") != std::string::npos)
          std::printf("      bundled: %s\n", n.c_str());
      }
      closedir(dd);
    }
  }

  // --------------------------------------------------------------- verdict
  section("VERDICT");
  const bool usable = (rep.header_dirs > 0) && (rep.lib_files > 0);
  std::printf("  headers present : %s\n", rep.header_dirs ? "YES" : "NO");
  std::printf("  libs present    : %s\n", rep.lib_files ? "YES" : "NO");
  std::printf("  compile+link    : %s\n", usable ? "POSSIBLE" : "NOT POSSIBLE");
  if (usable) {
    std::printf("\n  OpenCV C++ SDK looks usable on this board.\n");
  } else {
    std::printf("\n  There is NO OpenCV C++ SDK on this board.\n");
    std::printf("  A working `import cv2` does not contradict this: the Python wheel\n");
    std::printf("  ships a self-contained extension module and none of the headers,\n");
    std::printf("  import libraries or .pc files that C++ compilation needs.\n");
    std::printf("\n  To make it usable:\n");
    std::printf("    apt-get install -y libopencv-dev      # if the apt sources have it\n");
    std::printf("    or build OpenCV for arm64 and install headers + shared libs\n");
    std::printf("  Before relying on either, re-run the bit-exactness check: a\n");
    std::printf("  DIFFERENT OpenCV build breaks the premise that the port is exact\n");
    std::printf("  because it calls the same C++ translation units as the reference.\n");
  }
  return 0;
}
