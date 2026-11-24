/* stat_compat.c -- glibc < 2.33 的 `stat` 兼容垫片
 *
 * 背景：glibc 2.29~2.32 只导出 `__xstat(ver, path, buf)`，**不导出公开符号
 *       `stat`**（公开的 `stat` 只在 libc_nonshared.a 里以 hidden 形式存在）；
 *       而板卡上的 OpenCV 4.8（`libopencv_core.so.408`）引用的是**公开符号
 *       `stat`** ⇒ 用 buildroot 交叉工具链（glibc 2.29）链接**可执行文件**时，
 *       ld 会报：
 *           ld: .../libc_nonshared.a(stat.oS): hidden symbol `stat' in ...
 *              is referenced by DSO
 *           ld: final link failed: Bad value
 *       （链接**共享库**时不报，因为 `build_lib.sh` 带 `-Wl,--allow-shlib-undefined`；
 *         只有链接**可执行文件**才会撞上。）
 *
 * 修法：本文件定义公开符号 `stat`，内部转调 glibc 真实入口 `__xstat`
 *       （板卡上的 `/root/mosaic/libstatcompat.so` 用的是同一手法）。
 *       它被**静态编进每个目标可执行文件** ⇒ **无运行时依赖**
 *       （不需要 LD_PRELOAD，也不需要随包带任何 .so）。
 *       `__xstat` 的版本参数固定为 **1**（`_STAT_VER`）。
 *       链接到 glibc >= 2.33 的板子时语义同样正确（那边 `stat` 是真公开符号，
 *       本垫片只是多了一个未使用的定义，不影响行为）。
 *
 * 🔴【2026-11-23】**加了 glibc 版本护栏**：`__xstat` 在 glibc 2.33 起**被移除**
 *    ⇒ 用 glibc >= 2.33 的工具链编译本文件会得到一个**引用了不存在的符号**的
 *    .o（链接期报 `undefined reference to __xstat`）⇒ 目标板换了 glibc 大版本
 *    就链不出来。
 *    ⇒ 现在只在 **glibc < 2.33** 时才定义 `stat` 转发；>= 2.33 时本文件**编译出
 *      空目标文件**（那边 `stat` 本来就是公开符号，不需要垫片）。
 *    ⇒ 集成方可以用**任意 glibc 版本的工具链**编译这份源码，不用挑版本。
 *    ⚠️ 因此本垫片**故意不进**预编译的 `libmosaic_pipeline.a`（那是用固定
 *       工具链编的，会把 glibc 版本锁死）⇒ 静态链接时集成方**自己按目标板
 *       编这一份**，或者按目标板 glibc 版本决定要不要带。
 */
#include <features.h>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(__GLIBC__) && (__GLIBC__ < 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ < 33))
extern int __xstat(int ver, const char *path, struct stat *buf);

int stat(const char *path, struct stat *buf) { return __xstat(1, path, buf); }
#else
/* glibc >= 2.33（公开的 `stat` 就是真符号）/ 非 glibc（musl 等）⇒ 不需要垫片 */
typedef int mosaic_stat_compat_unused;
#endif
