#!/bin/bash

# Get the current hostname
hostname=$(hostname)

# Get CPU cores, sockets, memory, and GPU details
cpu_cores=$(grep -c processor /proc/cpuinfo)
sockets=$(lscpu | grep "Socket(s):" | awk '{print $2}')
cores_per_socket=$((cpu_cores / sockets))
total_memory=$(grep MemTotal /proc/meminfo | awk '{print int($2/1024)}')

# Detect GPU type and count
gpu_type=$(nvidia-smi -L | head -n 1 | \
  sed -n 's/.*: \(.*\) (UUID.*/\1/p' | sed 's/ /_/g')
gpu_count=$(nvidia-smi -L | wc -l)

# Backup original slurm.conf
cp /etc/slurm/slurm.conf /etc/slurm/slurm.conf.bak

# Update slurm.conf using awk
awk -v cm="$hostname" -v nn="$hostname" -v cpus="$cpu_cores" \
    -v socks="$sockets" -v cps="$cores_per_socket" \
    -v mem="$total_memory" -v gres="Gres=gpu:${gpu_type}:${gpu_count}" '
    /^ControlMachine=/ {$0="ControlMachine=" cm}
    /^NodeName=/ {
        $0="NodeName=" nn " CPUs=" cpus " Sockets=" socks \
        " CoresPerSocket=" cps " ThreadsPerCore=1 RealMemory=" mem \
        " " gres " State=IDLE"
    }
    /^PartitionName=/ {
        $0="PartitionName=main Nodes=" nn " Default=YES MaxTime=INFINITE State=UP"
    }
    {print}
' /etc/slurm/slurm.conf.bak > /etc/slurm/slurm.conf

echo "Updated slurm.conf successfully with the actual server values."