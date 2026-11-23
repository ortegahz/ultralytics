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
 */
#include <sys/stat.h>
#include <sys/types.h>

extern int __xstat(int ver, const char *path, struct stat *buf);

int stat(const char *path, struct stat *buf) { return __xstat(1, path, buf); }
