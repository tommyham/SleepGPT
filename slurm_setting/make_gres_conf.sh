#!/bin/bash

# gres.conf のバックアップを作成
if [ -f /etc/slurm/gres.conf ]; then
    mv /etc/slurm/gres.conf /etc/slurm/gres.conf.bak
fi

# ホスト名を取得
hostname=$(hostname)

# nvidia-smi -L から最初のGPUのタイプを取得
gpu_type=$(nvidia-smi -L | head -n 1 | sed -n 's/.*: \(.*\) (UUID.*/\1/p' | sed 's/ /_/g')

# nvidia-smi topo -m コマンドの出力から最初のヘッダ行を除外して取得
topo_output=$(nvidia-smi topo -m | tail -n +2)

# CPUアフィニティ情報の解析とgres.confフォーマットでの出力
{
echo "$topo_output" | awk -v hostname="$hostname" -v type="$gpu_type" '
BEGIN {
    min_gpu_id = 10000  # 十分に大きな値で初期化
    max_gpu_id = -1     # 十分に小さな値で初期化
    last_affinity = ""
}

/GPU[0-9]+\t/ {
    # GPU ID と CPUアフィニティを抽出
    gpu_id = substr($1, 4)
    cpu_affinity = $(NF-2)

    # 同じCPUアフィニティを持つGPUをグループ化
    if (cpu_affinity == last_affinity) {
        if (gpu_id < min_gpu_id) min_gpu_id = gpu_id
        if (gpu_id > max_gpu_id) max_gpu_id = gpu_id
    } else {
        if (last_affinity != "") {
            print "NodeName=" hostname " Name=gpu Type=" type " File=/dev/nvidia[" min_gpu_id "-" max_gpu_id "] COREs=" last_affinity
        }
        min_gpu_id = gpu_id
        max_gpu_id = gpu_id
        last_affinity = cpu_affinity
    }
}

END {
    # 最後のグループを表示
    if (last_affinity != "") {
        print "NodeName=" hostname " Name=gpu Type=" type " File=/dev/nvidia[" min_gpu_id "-" max_gpu_id "] COREs=" last_affinity
    }
}'
} > /etc/slurm/gres.conf

cat /etc/slurm/gres.conf
echo "New gres.conf has been created and old gres.conf has been backed up."