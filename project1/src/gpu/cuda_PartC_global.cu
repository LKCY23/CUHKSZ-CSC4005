//
// Created by Liu Yuxuan on 2024/9/11
// Modified from Zhong Yebin's PartB on 2023/9/16
//
// Email: yebinzhong@link.cuhk.edu.cn
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// CUDA implementation of bilateral filtering on JPEG image (Global Memory Version)
//

#include <iostream>
#include <cmath>
#include <chrono>
#include <cstring>
#include <cuda_runtime.h>

#include "../utils.hpp"

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
 * CUDA device function for bilateral filtering using global memory
 */
__device__ ColorValue d_bilateral_filter(const ColorValue* values, int pixel_id, int width,
                                       const float (*spatial_filter)[FILTERSIZE])
{
    float weight_sum = 0.0f;
    float value_sum = 0.0f;
    ColorValue center_value = values[pixel_id];
    
    // 预计算常量
    const float sigma_r_squared_2 = 2.0f * SIGMA_R * SIGMA_R;
    
    // 像 OpenACC PartC 一样简洁的循环展开写法，只计算强度权重
    // 中心点 (1,1)
    float intensity_weight = expf(-(0.0f) / sigma_r_squared_2);
    value_sum += values[pixel_id] * spatial_filter[1][1] * intensity_weight;
    weight_sum += spatial_filter[1][1] * intensity_weight;
    
    // 左 (1,0)
    intensity_weight = expf(-((center_value - values[pixel_id - 1]) * (center_value - values[pixel_id - 1])) / sigma_r_squared_2);
    value_sum += values[pixel_id - 1] * spatial_filter[1][0] * intensity_weight;
    weight_sum += spatial_filter[1][0] * intensity_weight;
    
    // 右 (1,2)
    intensity_weight = expf(-((center_value - values[pixel_id + 1]) * (center_value - values[pixel_id + 1])) / sigma_r_squared_2);
    value_sum += values[pixel_id + 1] * spatial_filter[1][2] * intensity_weight;
    weight_sum += spatial_filter[1][2] * intensity_weight;
    
    // 上 (0,1)
    intensity_weight = expf(-((center_value - values[pixel_id - width]) * (center_value - values[pixel_id - width])) / sigma_r_squared_2);
    value_sum += values[pixel_id - width] * spatial_filter[0][1] * intensity_weight;
    weight_sum += spatial_filter[0][1] * intensity_weight;
    
    // 左上 (0,0)
    intensity_weight = expf(-((center_value - values[pixel_id - width - 1]) * (center_value - values[pixel_id - width - 1])) / sigma_r_squared_2);
    value_sum += values[pixel_id - width - 1] * spatial_filter[0][0] * intensity_weight;
    weight_sum += spatial_filter[0][0] * intensity_weight;
    
    // 右上 (0,2)
    intensity_weight = expf(-((center_value - values[pixel_id - width + 1]) * (center_value - values[pixel_id - width + 1])) / sigma_r_squared_2);
    value_sum += values[pixel_id - width + 1] * spatial_filter[0][2] * intensity_weight;
    weight_sum += spatial_filter[0][2] * intensity_weight;
    
    // 下 (2,1)
    intensity_weight = expf(-((center_value - values[pixel_id + width]) * (center_value - values[pixel_id + width])) / sigma_r_squared_2);
    value_sum += values[pixel_id + width] * spatial_filter[2][1] * intensity_weight;
    weight_sum += spatial_filter[2][1] * intensity_weight;
    
    // 左下 (2,0)
    intensity_weight = expf(-((center_value - values[pixel_id + width - 1]) * (center_value - values[pixel_id + width - 1])) / sigma_r_squared_2);
    value_sum += values[pixel_id + width - 1] * spatial_filter[2][0] * intensity_weight;
    weight_sum += spatial_filter[2][0] * intensity_weight;
    
    // 右下 (2,2)
    intensity_weight = expf(-((center_value - values[pixel_id + width + 1]) * (center_value - values[pixel_id + width + 1])) / sigma_r_squared_2);
    value_sum += values[pixel_id + width + 1] * spatial_filter[2][2] * intensity_weight;
    weight_sum += spatial_filter[2][2] * intensity_weight;
    
    float result = (weight_sum > 0) ? (value_sum / weight_sum) : center_value;
    return d_clamp_pixel_value(result);
}

const float spatial_filter[FILTERSIZE][FILTERSIZE] = { // [Bilateral Filter Only] spatial weights
    {expf(-2.0f / (2.0f * SIGMA_D * SIGMA_D)), expf(-1.0f / (2.0f * SIGMA_D * SIGMA_D)), expf(-2.0f / (2.0f * SIGMA_D * SIGMA_D))},
    {expf(-1.0f / (2.0f * SIGMA_D * SIGMA_D)), expf(-0.0f / (2.0f * SIGMA_D * SIGMA_D)), expf(-1.0f / (2.0f * SIGMA_D * SIGMA_D))},
    {expf(-2.0f / (2.0f * SIGMA_D * SIGMA_D)), expf(-1.0f / (2.0f * SIGMA_D * SIGMA_D)), expf(-2.0f / (2.0f * SIGMA_D * SIGMA_D))}};

/**
 * CUDA kernel for bilateral filtering using global memory
 */
__global__ void bilateral_filter_global_kernel(const ColorValue* r_input, const ColorValue* g_input, const ColorValue* b_input,
                                              ColorValue* r_output, ColorValue* g_output, ColorValue* b_output,
                                              int width, int height,
                                              const float (*d_spatial_filter)[FILTERSIZE])
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 确保在有效范围内（边界不处理）
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1)
    {
        int index = y * width + x;
        
        // 对三个通道分别应用双边滤波
        r_output[index] = d_bilateral_filter(r_input, index, width, d_spatial_filter);
        g_output[index] = d_bilateral_filter(g_input, index, width, d_spatial_filter);
        b_output[index] = d_bilateral_filter(b_input, index, width, d_spatial_filter);
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
     * TODO: CUDA PartC Global Memory Version
     */
    
    // ========== CUDA 双边滤波实现（全局内存版本） ==========
    
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
    float (*d_spatial_filter)[FILTERSIZE];

    cudaMalloc((void**)&d_r_input, buffer_size * sizeof(ColorValue));
    cudaMalloc((void**)&d_g_input, buffer_size * sizeof(ColorValue));
    cudaMalloc((void**)&d_b_input, buffer_size * sizeof(ColorValue));
    cudaMalloc((void**)&d_r_output, buffer_size * sizeof(ColorValue));
    cudaMalloc((void**)&d_g_output, buffer_size * sizeof(ColorValue));
    cudaMalloc((void**)&d_b_output, buffer_size * sizeof(ColorValue));
    cudaMalloc((void**)&d_spatial_filter, FILTERSIZE * FILTERSIZE * sizeof(float));

    // 复制数据到 GPU
    cudaMemcpy(d_r_input, input_jpeg.r_values, buffer_size * sizeof(ColorValue), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g_input, input_jpeg.g_values, buffer_size * sizeof(ColorValue), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_input, input_jpeg.b_values, buffer_size * sizeof(ColorValue), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r_output, r_output, buffer_size * sizeof(ColorValue), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g_output, g_output, buffer_size * sizeof(ColorValue), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_output, b_output, buffer_size * sizeof(ColorValue), cudaMemcpyHostToDevice);
    cudaMemcpy(d_spatial_filter, spatial_filter, FILTERSIZE * FILTERSIZE * sizeof(float), cudaMemcpyHostToDevice);

    // 设置 CUDA 网格和块大小
    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // 性能测量
    cudaEvent_t start, stop;
    float gpuDuration;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 执行 GPU 计算（全局内存版本）
    cudaEventRecord(start, 0);
    bilateral_filter_global_kernel<<<gridDim, blockDim>>>(
        d_r_input, d_g_input, d_b_input,
        d_r_output, d_g_output, d_b_output,
        width, height, d_spatial_filter);
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
    cudaFree(d_spatial_filter);
    
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "GPU Execution Time (Global Memory): " << gpuDuration << " milliseconds" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
