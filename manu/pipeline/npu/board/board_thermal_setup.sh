#!/bin/bash
# 板卡测试环境重建（板卡重启后 /root 会丢失，需重跑本脚本）
#   /root/bts/  <- 二进制、模型、kernel、帧（全部落 eMMC 持久盘，不用 /tmp tmpfs）
set -e
BTS=/root/bts
mkdir -p "$BTS/kernels" "$BTS/frames"

# NFS（NFS 是手工挂的，重启后要重挂）
if ! mount | grep -q '/mnt/manu'; then
    mount -t nfs 192.168.0.115:/run/media/manu/SSD960-LOCAL/nfs /mnt/manu \
        -o nolock,soft,timeo=5,retrans=2 || true
fi

cp /mnt/manu/mosaic_sdk/out/mosaic_async_client_example "$BTS/"
cp /mnt/manu/mosaic_sdk/out/libmosaic_pipeline.so "$BTS/"
cp /mnt/manu/mosaic_sdk/out/rknn_mosaic_e2e "$BTS/"
cp /mnt/manu/ptq/models/p0_off_A_none.rknn "$BTS/"
cp /mnt/manu/mosaic_sdk/runtime/kernels/* "$BTS/kernels/"
chmod +x "$BTS/mosaic_async_client_example" "$BTS/rknn_mosaic_e2e"

# 帧：默认直接读 NFS，不做本地拷贝（COPY_FRAMES=1 才拷）
SEQ=VIDEO00007_19700101_001324
if [ "${COPY_FRAMES:-0}" = "1" ] && [ ! -f "$BTS/frames/$SEQ/.copy_done" ]; then
    cp -r "/mnt/manu/lq_ir_bmp/$SEQ" "$BTS/frames/" && touch "$BTS/frames/$SEQ/.copy_done"
fi

# 脚本：优先用 /mnt/manu/bts_scripts（host 磁盘，重启不丢）
if [ -d /mnt/manu/bts_scripts ]; then
    cp /mnt/manu/bts_scripts/board_thermal_* "$BTS/" 2>/dev/null || true
fi
chmod +x "$BTS"/*.sh "$BTS"/*.py 2>/dev/null || true
echo "=== setup done ==="
ls -la "$BTS"
md5sum "$BTS/mosaic_async_client_example" "$BTS/libmosaic_pipeline.so" "$BTS/p0_off_A_none.rknn"
