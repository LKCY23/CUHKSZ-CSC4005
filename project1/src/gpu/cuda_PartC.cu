//
// Created by Liu Yuxuan on 2024/9/11
// Modified from Zhong Yebin's PartB on 2023/9/16
//
// Email: yebinzhong@link.cuhk.edu.cn
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// CUDA implementation of bilateral filtering on JPEG image
//

#include <iostream>
#include <cmath>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>

#include "../utils.hpp"

// 常量内存：用于存储空间滤波器，提供更快的访问速度
__constant__ float d_spatial_filter_const[FILTERSIZE][FILTERSIZE];

/**
 * Demo kernel device function to clamp pixel value
 * 
 * You may mimic this to implement your own kernel device functions
 */
__device__ ColorValue d_clamp_pixel_value(float value)
{
    return value > 255 ? 255
           : value < 0 ? 0
                       : static_cast<ColorValue>(value);
}


/**
 * CUDA device function for bilateral filtering using shared memory
 * 使用共享内存进行双边滤波，避免全局内存访问
 */
__device__ ColorValue d_bilateral_filter_shared(ColorValue shared_data[34][34], int ty, int tx)
{
    const float sigma_r_sq_inv = -1.0f / (2.0f * SIGMA_R * SIGMA_R);
    
    float weight_sum = 0.0f;
    float value_sum = 0.0f;
    ColorValue center_value = shared_data[ty][tx];
    
    // 9个邻域点的处理，使用共享内存索引和常量内存
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            ColorValue neighbor = shared_data[ty + dy][tx + dx];
            float intensity_diff = center_value - neighbor;
            float intensity_weight = expf(intensity_diff * intensity_diff * sigma_r_sq_inv);
            float spatial_weight = d_spatial_filter_const[dy + 1][dx + 1];  // 使用常量内存
            
            value_sum += neighbor * spatial_weight * intensity_weight;
            weight_sum += spatial_weight * intensity_weight;
        }
    }
    
    float result = (weight_sum > 0) ? (value_sum / weight_sum) : center_value;
    return d_clamp_pixel_value(result);
}

const float spatial_filter[FILTERSIZE][FILTERSIZE] = { // [Bilateral Filter Only] spatial weights
    {expf(-2.0f / (2.0f * SIGMA_D * SIGMA_D)), expf(-1.0f / (2.0f * SIGMA_D * SIGMA_D)), expf(-2.0f / (2.0f * SIGMA_D * SIGMA_D))},
    {expf(-1.0f / (2.0f * SIGMA_D * SIGMA_D)), expf(-0.0f / (2.0f * SIGMA_D * SIGMA_D)), expf(-1.0f / (2.0f * SIGMA_D * SIGMA_D))},
    {expf(-2.0f / (2.0f * SIGMA_D * SIGMA_D)), expf(-1.0f / (2.0f * SIGMA_D * SIGMA_D)), expf(-2.0f / (2.0f * SIGMA_D * SIGMA_D))}};

/**
 * CUDA kernel for bilateral filtering using shared memory
 * 使用共享内存优化内存访问模式
 */
__global__ void bilateral_filter_shared_kernel(const ColorValue* r_input, const ColorValue* g_input, const ColorValue* b_input,
                                              ColorValue* r_output, ColorValue* g_output, ColorValue* b_output,
                                              int width, int height)
{
    // 共享内存：34x34 包含32x32工作区域 + 2像素边界
    __shared__ ColorValue r_shared[34][34];
    __shared__ ColorValue g_shared[34][34];
    __shared__ ColorValue b_shared[34][34];
    
    int tx = threadIdx.x;  // 0-31
    int ty = threadIdx.y;  // 0-31
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // 全局坐标
    int gx = bx * 32 + tx;
    int gy = by * 32 + ty;
    
    // 协作加载：每个线程加载一个像素到共享内存（包括边界）
    int load_x = gx - 1;  // 加载边界像素
    int load_y = gy - 1;
    
    if (load_x >= 0 && load_x < width && load_y >= 0 && load_y < height) {
        int load_idx = load_y * width + load_x;
        r_shared[ty+1][tx+1] = r_input[load_idx];
        g_shared[ty+1][tx+1] = g_input[load_idx];
        b_shared[ty+1][tx+1] = b_input[load_idx];
    } else {
        r_shared[ty+1][tx+1] = 0;
        g_shared[ty+1][tx+1] = 0;
        b_shared[ty+1][tx+1] = 0;
    }
    
    __syncthreads();  // 等待所有线程完成加载
    
    // 处理内部像素（避免边界）
    if (tx >= 1 && tx <= 30 && ty >= 1 && ty <= 30 && 
        gx >= 1 && gx < width-1 && gy >= 1 && gy < height-1) {
        
        int index = gy * width + gx;
        
        // 使用共享内存进行双边滤波
        r_output[index] = d_bilateral_filter_shared(r_shared, ty+1, tx+1);
        g_output[index] = d_bilateral_filter_shared(g_shared, ty+1, tx+1);
        b_output[index] = d_bilateral_filter_shared(b_shared, ty+1, tx+1);
    }
}


int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read input JPEG image in structure-of-array form
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    /**
     * TODO: CUDA PartC
     */
    
    // ========== CUDA 双边滤波实现 ==========
    
    // 获取图像尺寸
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    size_t buffer_size = width * height;
    
    // 分配输出缓冲区
    ColorValue* r_output = new ColorValue[buffer_size];
    ColorValue* g_output = new ColorValue[buffer_size];
    ColorValue* b_output = new ColorValue[buffer_size];
    
    // 初始化输出缓冲区（复制边界像素）
    // 复制第一行和最后一行
    memcpy(r_output, input_jpeg.r_values, width * sizeof(ColorValue));
    memcpy(g_output, input_jpeg.g_values, width * sizeof(ColorValue));
    memcpy(b_output, input_jpeg.b_values, width * sizeof(ColorValue));
    
    memcpy(r_output + (height-1) * width, 
           input_jpeg.r_values + (height-1) * width, 
           width * sizeof(ColorValue));
    memcpy(g_output + (height-1) * width, 
           input_jpeg.g_values + (height-1) * width, 
           width * sizeof(ColorValue));
    memcpy(b_output + (height-1) * width, 
           input_jpeg.b_values + (height-1) * width, 
           width * sizeof(ColorValue));
    
    // 复制边界列
    for (int row = 1; row < height - 1; ++row) {
        int row_offset = row * width;
        // 第一列
        r_output[row_offset] = input_jpeg.r_values[row_offset];
        g_output[row_offset] = input_jpeg.g_values[row_offset];
        b_output[row_offset] = input_jpeg.b_values[row_offset];
        // 最后一列
        r_output[row_offset + width - 1] = input_jpeg.r_values[row_offset + width - 1];
        g_output[row_offset + width - 1] = input_jpeg.g_values[row_offset + width - 1];
        b_output[row_offset + width - 1] = input_jpeg.b_values[row_offset + width - 1];
    }

    // 分配 GPU 内存
    ColorValue* d_r_input;
    ColorValue* d_g_input;
    ColorValue* d_b_input;
    ColorValue* d_r_output;
    ColorValue* d_g_output;
    ColorValue* d_b_output;

    cudaMalloc((void**)&d_r_input, buffer_size * sizeof(ColorValue));
    cudaMalloc((void**)&d_g_input, buffer_size * sizeof(ColorValue));
    cudaMalloc((void**)&d_b_input, buffer_size * sizeof(ColorValue));
    cudaMalloc((void**)&d_r_output, buffer_size * sizeof(ColorValue));
    cudaMalloc((void**)&d_g_output, buffer_size * sizeof(ColorValue));
    cudaMalloc((void**)&d_b_output, buffer_size * sizeof(ColorValue));

    // 复制数据到 GPU
    cudaMemcpy(d_r_input, input_jpeg.r_values, buffer_size * sizeof(ColorValue), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g_input, input_jpeg.g_values, buffer_size * sizeof(ColorValue), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_input, input_jpeg.b_values, buffer_size * sizeof(ColorValue), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_output, r_output, buffer_size * sizeof(ColorValue), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g_output, g_output, buffer_size * sizeof(ColorValue), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_output, b_output, buffer_size * sizeof(ColorValue), cudaMemcpyHostToDevice);
    // 将空间滤波器复制到常量内存
    cudaMemcpyToSymbol(d_spatial_filter_const, spatial_filter, FILTERSIZE * FILTERSIZE * sizeof(float));

    // 设置 CUDA 网格和块大小（使用共享内存优化的块大小）
    dim3 blockDim(32, 32);  // 32x32 = 1024 threads per block
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // 性能测量
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 执行 GPU 计算（使用共享内存优化版本）
    cudaEventRecord(start, 0);
    bilateral_filter_shared_kernel<<<gridDim, blockDim>>>(
        d_r_input, d_g_input, d_b_input,
        d_r_output, d_g_output, d_b_output,
        width, height);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // 获取执行时间
    cudaEventElapsedTime(&gpuDuration, start, stop);
    
    // 将结果从 GPU 复制回 CPU
    cudaMemcpy(r_output, d_r_output, buffer_size * sizeof(ColorValue), cudaMemcpyDeviceToHost);
    cudaMemcpy(g_output, d_g_output, buffer_size * sizeof(ColorValue), cudaMemcpyDeviceToHost);
    cudaMemcpy(b_output, d_b_output, buffer_size * sizeof(ColorValue), cudaMemcpyDeviceToHost);

    // 保存输出 JPEG 图像
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    JpegSOA output_jpeg{r_output, g_output, b_output, input_jpeg.width, 
                       input_jpeg.height, input_jpeg.num_channels, input_jpeg.color_space};
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }
    
    // 清理内存
    delete[] r_output;
    delete[] g_output;
    delete[] b_output;
    
    // 释放 GPU 内存
    cudaFree(d_r_input);
    cudaFree(d_g_input);
    cudaFree(d_b_input);
    cudaFree(d_r_output);
    cudaFree(d_g_output);
    cudaFree(d_b_output);
    
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time: " << gpuDuration << " milliseconds" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
