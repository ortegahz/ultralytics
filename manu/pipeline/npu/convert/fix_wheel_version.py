#!/usr/bin/env python3
"""Repack the Rockchip rknn_toolkit2 wheel with a PEP 440 compliant version.

The SDK ships `rknn_toolkit2-1.4.0_22dcfef4-cp38-cp38-linux_x86_64.whl`, whose
filename AND dist-info metadata both carry `1.4.0_22dcfef4` / `1.4.0-22dcfef4`.
Neither parses as a PEP 440 version, so pip refuses the package outright:

    ERROR: Invalid requirement: 'rknn-toolkit2==1.4.0-22dcfef4'
    packaging.version.InvalidVersion: Invalid version: '1.4.0-22dcfef4'

The upstream build stamps a git hash as a *pre-release* segment, which PEP 440
only permits in the `+local` form. Rewriting it to `1.4.0+22dcfef4` keeps the
build identity (the hash survives) while making the package installable.

The SDK tree is never modified; the rewritten wheel is written to the work dir.
"""
import os
import re
import shutil
import sys
import zipfile

SRC = sys.argv[1]
DST = sys.argv[2]
NEW_VERSION = "1.4.0+22dcfef4"
OLD_NAME_RE = re.compile(r"^rknn_toolkit2-([0-9A-Za-z_.+-]+)\.dist-info$")

os.makedirs(os.path.dirname(DST), exist_ok=True)
if os.path.exists(DST):
    os.remove(DST)

renames = {}
version_lines_fixed = 0
with zipfile.ZipFile(SRC) as zin, zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        name = item.filename
        base = os.path.basename(name)
        parent = os.path.dirname(name)

        m = OLD_NAME_RE.match(base)
        if m:
            base = "rknn_toolkit2-%s.dist-info" % NEW_VERSION
            name = os.path.join(parent, base) if parent else base
            renames[base] = True

        data = zin.read(item.filename)
        if item.filename.endswith(".dist-info/METADATA"):
            text = data.decode("utf-8", "replace")
            new_text, n = re.subn(r"^Version:.*$", "Version: %s" % NEW_VERSION, text,
                                  count=1, flags=re.MULTILINE)
            if n:
                version_lines_fixed += 1
            data = new_text.encode("utf-8")

        zout.writestr(name, data)

print("rewrote METADATA Version lines:", version_lines_fixed)
print("dist-info dirs renamed:", len(renames))
print("wrote:", DST)
with zipfile.ZipFile(DST) as z:
    print("sanity, dist-info entries:", [n for n in z.namelist() if n.endswith(".dist-info/METADATA")])