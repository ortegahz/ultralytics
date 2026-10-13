#!/usr/bin/env python3
"""Clear the executable bit from PT_GNU_STACK in ELF shared objects.

Some prebuilt wheels (onnxruntime >= 1.10 among them) ship with a PT_GNU_STACK
segment marked RWX. On any host that refuses executable stacks the loader aborts
with

    ImportError: ... cannot enable executable stack as shared object requires:
    Invalid argument

which surfaces as a confusing ImportError rather than a security-policy message.

`patchelf --clear-execstack` is the usual remedy; this does the same thing
without the dependency by editing the program header in place:

  Elf64_Phdr is 56 bytes, p_type at +0, p_flags at +4, PT_GNU_STACK = 0x6474e551.
  PF_X is bit 0x1 in p_flags.

The edit is a single byte in the file, so nothing is re-laid-out and the rest of
the object is untouched. Files are rewritten atomically (tmp + rename).

Usage: clear_execstack.py <path> [<path> ...]
       clear_execstack.py --scan <dir>
"""
import os
import struct
import sys

PT_GNU_STACK = 0x6474E551
PF_X = 0x1
ELFCLASS64 = 2
PT_GNU_STACK_32 = 0x6474E551  # same value on 32-bit


def scan(paths):
    """Return [(path, 'RWX'|'none', 'fixed'|'skipped')] for every ELF SO found."""
    out = []
    for root in paths:
        if os.path.isfile(root):
            files = [root]
        else:
            files = []
            for dp, _dn, fn in os.walk(root):
                for f in fn:
                    if f.endswith((".so", ".so.1")) or ".so." in f:
                        files.append(os.path.join(dp, f))
        for p in files:
            try:
                state = fix(p)
            except Exception as e:  # not ELF, unreadable, etc.
                state = "skip(%s)" % type(e).__name__
            if state:
                out.append((p, state))
    return out


def fix(path):
    with open(path, "rb") as f:
        data = bytearray(f.read())

    if len(data) < 64 or bytes(data[:4]) != b"\x7fELF":
        return None

    is64 = data[4] == ELFCLASS64
    e_phoff = struct.unpack_from("<Q", data, 0x20)[0] if is64 else struct.unpack_from("<I", data, 0x1C)[0]
    e_phentsize = struct.unpack_from("<H", data, 0x36 if is64 else 0x2A)[0]
    e_phnum = struct.unpack_from("<H", data, 0x38 if is64 else 0x2C)[0]

    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        if off + e_phentsize > len(data):
            break
        p_type = struct.unpack_from("<I", data, off)[0]
        if p_type != PT_GNU_STACK:
            continue
        p_flags = struct.unpack_from("<I", data, off + 4)[0]
        if not (p_flags & PF_X):
            return "already-nx"
        struct.pack_into("<I", data, off + 4, p_flags & ~PF_X)

        tmp = path + ".nxtmp"
        with open(tmp, "wb") as g:
            g.write(data)
            os.chmod(tmp, os.stat(path).st_mode)
        os.replace(tmp, path)
        return "cleared"
    return None


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args[0] == "--scan":
        for p, s in scan(args[1:]):
            print("%-12s %s" % (s, p))
        return
    n = 0
    for p in args:
        s = fix(p)
        if s:
            n += 1
            print("%-12s %s" % (s, p))
    print("\n%d file(s) touched" % n)


if __name__ == "__main__":
    main()