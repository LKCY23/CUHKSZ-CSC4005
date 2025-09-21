#!/bin/bash
# 本地运行版本的PartA测试脚本
# 基于sbatch_PartA.sh修改，移除SLURM依赖

# 设置环境变量（更新为你的NVIDIA HPC SDK版本）
export TRITON_PTXAS_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/bin/cuobjdump
export TRITON_NVDISASM_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/bin/nvdisasm

# 获取当前目录
CURRENT_DIR=$(pwd)/src/scripts
echo "Current directory: ${CURRENT_DIR}"

# 检查可执行文件是否存在
if [ ! -f "${CURRENT_DIR}/../../build/src/cpu/sequential_PartA" ]; then
    echo "Error: 可执行文件不存在，请先编译项目"
    echo "运行: cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

echo "开始PartA性能测试..."
echo "=========================================="

# Sequential PartA
echo "Sequential PartA (Optimized with -O2)"
time ${CURRENT_DIR}/../../build/src/cpu/sequential_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
echo ""

# Vectorization PartA
echo "Vectorization PartA (Optimized with -O2)"
time ${CURRENT_DIR}/../../build/src/cpu/vectorize_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
echo ""

# SIMD PartA [Optional]
echo "SIMD(AVX2) PartA (Optimized with -O2)"
time ${CURRENT_DIR}/../../build/src/cpu/simd_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
echo ""

# MPI PartA
echo "MPI PartA (Optimized with -O2)"
for num_processes in 1 2 4 8
do
  echo "Number of processes: $num_processes"
  time mpirun -np $num_processes ${CURRENT_DIR}/../../build/src/cpu/mpi_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
  echo ""
done

# Pthread PartA
echo "Pthread PartA (Optimized with -O2)"
for num_cores in 1 2 4 8
do
  echo "Number of cores: $num_cores"
  time ${CURRENT_DIR}/../../build/src/cpu/pthread_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg ${num_cores}
  echo ""
done

# OpenMP PartA
echo "OpenMP PartA (Optimized with -O2)"
for num_cores in 1 2 4 8
do
  echo "Number of cores: $num_cores"
  export OMP_NUM_THREADS=$num_cores
  time ${CURRENT_DIR}/../../build/src/cpu/openmp_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
  echo ""
done

# CUDA PartA
echo "CUDA PartA"
time ${CURRENT_DIR}/../../build/src/gpu/cuda_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
echo ""

# OpenACC PartA
echo "OpenACC PartA"
time ${CURRENT_DIR}/../../build/src/gpu/openacc_PartA ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
echo ""

# Triton PartA
echo "Triton PartA"
time python3 ${CURRENT_DIR}/../gpu/triton_PartA.py ${CURRENT_DIR}/../../images/20K-RGB.jpg ${CURRENT_DIR}/../../images/20K-Gray.jpg
echo ""

echo "=========================================="
echo "PartA性能测试完成！"
