// mosaic_paths.h —— 板卡无关的「默认路径」解析（SDK 自带的 demo / 探针用）
//
// ---------------------------------------------------------------------------
// 为什么需要它（2026-11-25 适配 Orange Pi 5 / Debian 12 时踩出来的）
// ---------------------------------------------------------------------------
//   这批 demo 早期把路径**硬编码**成**旧板的 NFS 挂载点** `/mnt/manu/...`：
//     frames : /mnt/manu/lq_ir_bmp/VIDEO00007_19700101_001324
//     kernels: /mnt/manu/mosaic_sdk/runtime/kernels
//   换到实际部署板（Orange Pi 5，板上**没有** /mnt/manu 这个挂载点）之后，
//   demo 直接 `cannot open .../warp_median_fused.cl`（[OCL FATAL]）跑不起来。
//
//   现在统一改成**三级回落**（谁先命中用谁）：
//     ① 环境变量 —— 部署方在启动脚本里设一次，全局生效
//     ② **交付包内** —— <可执行文件所在目录>/../runtime/kernels、../models/...
//        （可执行文件在 <pkg>/out/ 下 ⇒ 包根就是它的上一级）
//        ⇒ **整包丢到板卡任意目录都能直接跑，不依赖任何绝对路径**
//     ③ 已知板卡部署路径 —— 新板 /home/orangepi/runtime/...、旧板 /root/...、
//        旧 NFS 挂载 /mnt/manu/...
//
//   ⚠️ **命令行参数优先级永远高于这里的默认值**（env < 包内 < 板卡路径 < 硬编码兜底）。
//
// ---------------------------------------------------------------------------
// 约定的环境变量
// ---------------------------------------------------------------------------
//   MOSAIC_KERNEL_DIR   内核（.cl）目录
//   MOSAIC_MODEL        RKNN 模型（.rknn）路径
//   MOSAIC_RAW_ROOT     帧根目录（下面再拼序列名）
//   MOSAIC_FRAMES_DIR   直接指定帧目录（含序列名那一层），优先级最高
// ---------------------------------------------------------------------------
#pragma once

#include <cstdlib>
#include <cstring>
#include <string>

#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mosaic_paths {

inline bool Exists(const std::string& p)
{
  struct stat st;
  return !p.empty() && ::stat(p.c_str(), &st) == 0;
}

inline bool IsDir(const std::string& p)
{
  struct stat st;
  return !p.empty() && ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

// 可执行文件所在目录（读 /proc/self/exe；读不到就返回空串）
inline std::string ExeDir()
{
  char buf[PATH_MAX];
  const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
  if (n <= 0) return std::string();
  buf[n] = '\0';
  const std::string s(buf);
  const size_t pos = s.find_last_of('/');
  return pos == std::string::npos ? std::string() : s.substr(0, pos);
}

// 交付包根目录：可执行文件在 <pkg>/out/ 下 ⇒ 包根 = 它的上一级（**只剥一级**）
inline std::string PackageDir()
{
  const std::string d = ExeDir();
  if (d.empty()) return d;
  const size_t pos = d.find_last_of('/');
  if (pos == std::string::npos) return d;
  const std::string parent = d.substr(0, pos);
  return parent;
}

// 环境变量优先；未设置/为空则用 fallback
inline std::string EnvOr(const char* name, const std::string& fallback)
{
  const char* v = ::getenv(name);
  return (v && v[0]) ? std::string(v) : fallback;
}

// ---- 内核目录（OpenCL .cl）------------------------------------------------
inline std::string KernelDir()
{
  const std::string env = EnvOr("MOSAIC_KERNEL_DIR", "");
  if (!env.empty()) return env;

  const std::string pkg = PackageDir();
  if (!pkg.empty() && IsDir(pkg + "/runtime/kernels")) return pkg + "/runtime/kernels";

  const char* cands[] = {
      "/home/orangepi/runtime/kernels",       // 实际部署板（Orange Pi 5）
      "/root/kernels",                        // 旧板
      "/root/bts/kernels",                    // 旧板测试台
      "/mnt/manu/mosaic_sdk/runtime/kernels", // 旧 NFS 挂载（历史兜底）
  };
  for (const char* c : cands) {
    if (IsDir(c)) return c;
  }
  return "/mnt/manu/mosaic_sdk/runtime/kernels";
}

// ---- RKNN 模型 -------------------------------------------------------------
// ⚠️ 保持「板卡 tmpfs 优先」的老口径：交付目录在 NFS **soft 挂载**上时，
//    6.3 MB 的模型直读会被读坏 ⇒ 历史上要求先
//    `cat <交付目录>/models/p0_off_A_none.rknn > /tmp/p0_off_A_none.rknn`。
//    所以 /tmp 里的副本优先于包内副本。
inline std::string ModelPath()
{
  const std::string env = EnvOr("MOSAIC_MODEL", "");
  if (!env.empty()) return env;

  if (Exists("/tmp/p0_off_A_none.rknn")) return "/tmp/p0_off_A_none.rknn";

  const std::string pkg = PackageDir();
  if (!pkg.empty() && Exists(pkg + "/models/p0_off_A_none.rknn")) {
    return pkg + "/models/p0_off_A_none.rknn";
  }
  return "/tmp/p0_off_A_none.rknn";
}

// ---- 帧根目录（下面再拼序列名）--------------------------------------------
inline std::string FramesRoot()
{
  const std::string env = EnvOr("MOSAIC_RAW_ROOT", "");
  if (!env.empty()) return env;

  const char* cands[] = {
      "/root/bts/frames",       // 实际部署板的测试台
      "/mnt/manu/lq_ir_bmp",    // 旧 NFS 挂载（历史兜底）
  };
  for (const char* c : cands) {
    if (IsDir(c)) return c;
  }
  return "/mnt/manu/lq_ir_bmp";
}

}  // namespace mosaic_paths
