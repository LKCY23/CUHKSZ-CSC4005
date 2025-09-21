#!/bin/bash
# 本地运行版本的PartB测试脚本
# 基于sbatch_PartB.sh修改，移除SLURM依赖

# 设置环境变量（更新为你的NVIDIA HPC SDK版本）
export TRITON_PTXAS_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/bin/cuobjdump
export TRITON_NVDISASM_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/bin/nvdisasm

# 获取当前目录
CURRENT_DIR=$(pwd)/src/scripts
echo "Current directory: ${CURRENT_DIR}"

# 检查可执行文件是否存在
if [ ! -f "${CURRENT_DIR}/../../build/src/cpu/sequential_PartB" ]; then
    echo "Error: 可执行文件不存在，请先编译项目"
    echo "运行: cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

echo "开始PartB性能测试..."
echo "=========================================="

# Sequential PartB (Array-of-Structure)
echo "Sequential PartB (Array-of-Structure) (Optimized with -O2)"
time ${CURRENT_DIR}/../../build/src/cpu/sequential_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg
echo ""

# Sequential PartB (Structure-of-Array)
echo "Sequential PartB (Structure-of-Array) (Optimized with -O2)"
time ${CURRENT_DIR}/../../build/src/cpu/sequential_PartB_soa ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg
echo ""

# Vectorization PartB
echo "Vectorization PartB (Optimized with -O2)"
time ${CURRENT_DIR}/../../build/src/cpu/vectorize_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg
echo ""

# SIMD PartB
echo "SIMD(AVX2) PartB (Optimized with -O2)"
time ${CURRENT_DIR}/../../build/src/cpu/simd_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg
echo ""

# MPI PartB
echo "MPI PartB (Optimized with -O2)"
for num_processes in 1 2 4 8
do
  echo "Number of processes: $num_processes"
  time mpirun -np $num_processes ${CURRENT_DIR}/../../build/src/cpu/mpi_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg
  echo ""
done

# Pthread PartB
echo "Pthread PartB (Optimized with -O2)"
for num_cores in 1 2 4 8
do
  echo "Number of cores: $num_cores"
  time ${CURRENT_DIR}/../../build/src/cpu/pthread_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg ${num_cores}
  echo ""
done

# OpenMP PartB
echo "OpenMP PartB (Optimized with -O2)"
for num_cores in 1 2 4 8
do
  echo "Number of cores: $num_cores"
  export OMP_NUM_THREADS=$num_cores
  time ${CURRENT_DIR}/../../build/src/cpu/openmp_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg ${num_cores}
  echo ""
done

# CUDA PartB
echo "CUDA PartB"
time ${CURRENT_DIR}/../../build/src/gpu/cuda_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg
echo ""

# OpenACC PartB
echo "OpenACC PartB"
time ${CURRENT_DIR}/../../build/src/gpu/openacc_PartB ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg
echo ""

# Triton PartB
if [ "$TRITON_AVAILABLE" = true ]; then
    echo "Triton PartB"
    time python3 ${CURRENT_DIR}/../gpu/triton_PartB.py ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Smooth.jpg ${CURRENT_DIR}/../../images/time_PartB.png
else
    echo "Triton PartB - SKIPPED (Triton not available)"
fi
echo ""

echo "=========================================="
echo "PartB性能测试完成！"
