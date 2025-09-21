#!/bin/bash
# 本地运行版本的PartC测试脚本
# 基于sbatch_PartC.sh修改，移除SLURM依赖

# 设置环境变量（更新为你的NVIDIA HPC SDK版本）
export TRITON_PTXAS_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/bin/ptxas
export TRITON_CUOBJDUMP_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/bin/cuobjdump
export TRITON_NVDISASM_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/24.11/cuda/12.6/bin/nvdisasm

# 获取当前目录
CURRENT_DIR=$(pwd)/src/scripts
echo "Current directory: ${CURRENT_DIR}"

# 检查可执行文件是否存在
if [ ! -f "${CURRENT_DIR}/../../build/src/cpu/sequential_PartC_aos" ]; then
    echo "Error: 可执行文件不存在，请先编译项目"
    echo "运行: cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

echo "开始PartC性能测试..."
echo "=========================================="

# Sequential PartC (Array-of-Structure)
echo "Sequential PartC Array-of-Structure (Optimized with -O2)"
time ${CURRENT_DIR}/../../build/src/cpu/sequential_PartC_aos ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral.jpg
echo ""

# Sequential PartC (Structure-of-Array)
echo "Sequential PartC Structure-of-Array (Optimized with -O2)"
time ${CURRENT_DIR}/../../build/src/cpu/sequential_PartC_soa ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral.jpg
echo ""

# Vectorization PartC
echo "Vectorization PartC (Optimized with -O2)"
time ${CURRENT_DIR}/../../build/src/cpu/vectorize_PartC ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral.jpg
echo ""

# SIMD PartC
echo "SIMD(AVX2) PartC (Optimized with -O2)"
time ${CURRENT_DIR}/../../build/src/cpu/simd_PartC ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral.jpg
echo ""

# MPI PartC
echo "MPI PartC (Optimized with -O2)"
for num_processes in 1 2 4 8
do
  echo "Number of processes: $num_processes"
  time mpirun -np $num_processes ${CURRENT_DIR}/../../build/src/cpu/mpi_PartC ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral.jpg
  echo ""
done

# Pthread PartC
echo "Pthread PartC (Optimized with -O2)"
for num_cores in 1 2 4 8
do
  echo "Number of cores: $num_cores"
  time ${CURRENT_DIR}/../../build/src/cpu/pthread_PartC ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral.jpg ${num_cores}
  echo ""
done

# OpenMP PartC
echo "OpenMP PartC (Optimized with -O2)"
for num_cores in 1 2 4 8
do
  echo "Number of cores: $num_cores"
  export OMP_NUM_THREADS=$num_cores
  time ${CURRENT_DIR}/../../build/src/cpu/openmp_PartC ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral.jpg ${num_cores}
  echo ""
done

# CUDA PartC
echo "CUDA PartC"
time ${CURRENT_DIR}/../../build/src/gpu/cuda_PartC ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral.jpg
echo ""

# OpenACC PartC
echo "OpenACC PartC"
time ${CURRENT_DIR}/../../build/src/gpu/openacc_PartC ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral.jpg
echo ""

# Triton PartC
if [ "$TRITON_AVAILABLE" = true ]; then
    echo "Triton PartC"
    time python3 ${CURRENT_DIR}/../gpu/triton_PartC.py ${CURRENT_DIR}/../../images/4K-RGB.jpg ${CURRENT_DIR}/../../images/4K-Bilateral.jpg
else
    echo "Triton PartC - SKIPPED (Triton not available)"
fi
echo ""

echo "=========================================="
echo "PartC性能测试完成！"
