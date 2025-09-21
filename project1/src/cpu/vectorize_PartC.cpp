//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// Sequential implementation of converting a JPEG from RGB to gray
// (Strcture-of-Array)
//

#include <memory.h>

#include <chrono>
#include <cmath>
#include <iostream>

#include "../utils.hpp"

/**
 * Perform bilateral filter on a single Pixel (Structure-of-Array form)
 *
 * @return filtered pixel value
 */
inline ColorValue bilateral_filter_vectorize(
    const ColorValue* const __restrict__ values, const int row, const int col,
    const int width)
{
    const float w_border = expf(-0.5f / (SIGMA_D * SIGMA_D));
    const float w_corner = expf(-1.0f / (SIGMA_D * SIGMA_D));
    const float sigma_r_sq_inv = -0.5f / (SIGMA_R * SIGMA_R);

    // 获取3x3邻域的像素值，利用SOA数据结构的连续内存访问
    const ColorValue* const __restrict__ row_prev = values + (row - 1) * width;
    const ColorValue* const __restrict__ row_curr = values + row * width;
    const ColorValue* const __restrict__ row_next = values + (row + 1) * width;
    
    // 直接指针访问，避免重复计算索引
    // 利用SOA数据结构的连续内存特性，编译器可以更好地优化
    const ColorValue value_11 = row_prev[col - 1];
    const ColorValue value_12 = row_prev[col];
    const ColorValue value_13 = row_prev[col + 1];
    const ColorValue value_21 = row_curr[col - 1];
    const ColorValue value_22 = row_curr[col];  // 中心像素
    const ColorValue value_23 = row_curr[col + 1];
    const ColorValue value_31 = row_next[col - 1];
    const ColorValue value_32 = row_next[col];
    const ColorValue value_33 = row_next[col + 1];
    
    // 计算强度权重 - 向量化友好的计算
    const ColorValue center_value = value_22;
    
    // 预计算强度差值的平方
    const float diff_11 = center_value - value_11;
    const float diff_12 = center_value - value_12;
    const float diff_13 = center_value - value_13;
    const float diff_21 = center_value - value_21;
    const float diff_23 = center_value - value_23;
    const float diff_31 = center_value - value_31;
    const float diff_32 = center_value - value_32;
    const float diff_33 = center_value - value_33;
    
    // 计算权重 - 使用预计算的常量
    const float w_11 = w_corner * expf(diff_11 * diff_11 * sigma_r_sq_inv);
    const float w_12 = w_border * expf(diff_12 * diff_12 * sigma_r_sq_inv);
    const float w_13 = w_corner * expf(diff_13 * diff_13 * sigma_r_sq_inv);
    const float w_21 = w_border * expf(diff_21 * diff_21 * sigma_r_sq_inv);
    const float w_22 = 1.0f;  // 中心像素权重为1
    const float w_23 = w_border * expf(diff_23 * diff_23 * sigma_r_sq_inv);
    const float w_31 = w_corner * expf(diff_31 * diff_31 * sigma_r_sq_inv);
    const float w_32 = w_border * expf(diff_32 * diff_32 * sigma_r_sq_inv);
    const float w_33 = w_corner * expf(diff_33 * diff_33 * sigma_r_sq_inv);
    
    // 计算权重总和
    const float sum_weights = w_11 + w_12 + w_13 + w_21 + w_22 + w_23 + w_31 + w_32 + w_33;
    
    // 计算加权平均值
    const float filtered_value = (w_11 * value_11 + w_12 * value_12 + w_13 * value_13 +
                                  w_21 * value_21 + w_22 * center_value + w_23 * value_23 +
                                  w_31 * value_31 + w_32 * value_32 + w_33 * value_33) / sum_weights;

    return clamp_pixel_value(filtered_value);
}

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read JPEG File
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    // Apply the filter to the image
    const int width = input_jpeg.width;
    const int height = input_jpeg.height;
    const int num_channels = input_jpeg.num_channels;
    auto output_r_values = new ColorValue[width * height];
    auto output_g_values = new ColorValue[width * height];
    auto output_b_values = new ColorValue[width * height];
    JpegSOA output_jpeg{
        output_r_values, output_g_values, output_b_values,       width,
        height,          num_channels,    input_jpeg.color_space};
    auto start_time = std::chrono::high_resolution_clock::now();
    ColorValue* __restrict__ buf_r = input_jpeg.get_channel(0);
    ColorValue* __restrict__ buf_g = input_jpeg.get_channel(1);
    ColorValue* __restrict__ buf_b = input_jpeg.get_channel(2);
    ColorValue* output_r = output_jpeg.r_values;
    ColorValue* output_g = output_jpeg.g_values;
    ColorValue* output_b = output_jpeg.b_values;
    /* Pixels in the boundary can be ignored in this assignment */
    for (int row = 1; row < height - 1; ++row)
    {
        // 利用SOA数据结构特性：同一行的像素在内存中连续存储
        // 使用ivdep pragma告诉编译器可以进行向量化
        // 添加循环展开优化，更好地利用缓存和流水线
        const int base_index = row * width;

#pragma GCC ivdep
        // 缓存行优化：一次处理15个像素，充分利用64字节缓存行
        // 完全手动展开，最大化性能
        
        for (int col = 1; col < width - 16; col += 15)
        {
            // 计算起始索引一次，后续都基于加法
            const int index0 = base_index + col;
            const int index1 = index0 + 1;
            const int index2 = index0 + 2;
            const int index3 = index0 + 3;
            const int index4 = index0 + 4;
            const int index5 = index0 + 5;
            const int index6 = index0 + 6;
            const int index7 = index0 + 7;
            const int index8 = index0 + 8;
            const int index9 = index0 + 9;
            const int index10 = index0 + 10;
            const int index11 = index0 + 11;
            const int index12 = index0 + 12;
            const int index13 = index0 + 13;
            const int index14 = index0 + 14;
            
            // 手动展开15次处理
            // 0
            output_r[index0] = bilateral_filter_vectorize(buf_r, row, col, width);
            output_g[index0] = bilateral_filter_vectorize(buf_g, row, col, width);
            output_b[index0] = bilateral_filter_vectorize(buf_b, row, col, width);
            
            // 1
            output_r[index1] = bilateral_filter_vectorize(buf_r, row, col+1, width);
            output_g[index1] = bilateral_filter_vectorize(buf_g, row, col+1, width);
            output_b[index1] = bilateral_filter_vectorize(buf_b, row, col+1, width);
            
            // 2
            output_r[index2] = bilateral_filter_vectorize(buf_r, row, col+2, width);
            output_g[index2] = bilateral_filter_vectorize(buf_g, row, col+2, width);
            output_b[index2] = bilateral_filter_vectorize(buf_b, row, col+2, width);
            
            // 3
            output_r[index3] = bilateral_filter_vectorize(buf_r, row, col+3, width);
            output_g[index3] = bilateral_filter_vectorize(buf_g, row, col+3, width);
            output_b[index3] = bilateral_filter_vectorize(buf_b, row, col+3, width);
            
            // 4
            output_r[index4] = bilateral_filter_vectorize(buf_r, row, col+4, width);
            output_g[index4] = bilateral_filter_vectorize(buf_g, row, col+4, width);
            output_b[index4] = bilateral_filter_vectorize(buf_b, row, col+4, width);
            
            // 5
            output_r[index5] = bilateral_filter_vectorize(buf_r, row, col+5, width);
            output_g[index5] = bilateral_filter_vectorize(buf_g, row, col+5, width);
            output_b[index5] = bilateral_filter_vectorize(buf_b, row, col+5, width);
            
            // 6
            output_r[index6] = bilateral_filter_vectorize(buf_r, row, col+6, width);
            output_g[index6] = bilateral_filter_vectorize(buf_g, row, col+6, width);
            output_b[index6] = bilateral_filter_vectorize(buf_b, row, col+6, width);
            
            // 7
            output_r[index7] = bilateral_filter_vectorize(buf_r, row, col+7, width);
            output_g[index7] = bilateral_filter_vectorize(buf_g, row, col+7, width);
            output_b[index7] = bilateral_filter_vectorize(buf_b, row, col+7, width);
            
            // 8
            output_r[index8] = bilateral_filter_vectorize(buf_r, row, col+8, width);
            output_g[index8] = bilateral_filter_vectorize(buf_g, row, col+8, width);
            output_b[index8] = bilateral_filter_vectorize(buf_b, row, col+8, width);
            
            // 9
            output_r[index9] = bilateral_filter_vectorize(buf_r, row, col+9, width);
            output_g[index9] = bilateral_filter_vectorize(buf_g, row, col+9, width);
            output_b[index9] = bilateral_filter_vectorize(buf_b, row, col+9, width);
            
            // 10
            output_r[index10] = bilateral_filter_vectorize(buf_r, row, col+10, width);
            output_g[index10] = bilateral_filter_vectorize(buf_g, row, col+10, width);
            output_b[index10] = bilateral_filter_vectorize(buf_b, row, col+10, width);
            
            // 11
            output_r[index11] = bilateral_filter_vectorize(buf_r, row, col+11, width);
            output_g[index11] = bilateral_filter_vectorize(buf_g, row, col+11, width);
            output_b[index11] = bilateral_filter_vectorize(buf_b, row, col+11, width);
            
            // 12
            output_r[index12] = bilateral_filter_vectorize(buf_r, row, col+12, width);
            output_g[index12] = bilateral_filter_vectorize(buf_g, row, col+12, width);
            output_b[index12] = bilateral_filter_vectorize(buf_b, row, col+12, width);
            
            // 13
            output_r[index13] = bilateral_filter_vectorize(buf_r, row, col+13, width);
            output_g[index13] = bilateral_filter_vectorize(buf_g, row, col+13, width);
            output_b[index13] = bilateral_filter_vectorize(buf_b, row, col+13, width);
            
            // 14
            output_r[index14] = bilateral_filter_vectorize(buf_r, row, col+14, width);
            output_g[index14] = bilateral_filter_vectorize(buf_g, row, col+14, width);
            output_b[index14] = bilateral_filter_vectorize(buf_b, row, col+14, width);
            
            // 预读下一个缓存行的数据
            const int prefetch_index = base_index + col + 16;
            if (prefetch_index < base_index + width - 1)
            {
                __builtin_prefetch(&buf_r[prefetch_index], 0, 3);
                __builtin_prefetch(&buf_g[prefetch_index], 0, 3);
                __builtin_prefetch(&buf_b[prefetch_index], 0, 3);
            }
        }
        
        // 处理剩余的像素
        for (int col = ((width - 1) / 15) * 15 + 1; col < width - 1; ++col)
        {
            const int index = base_index + col;
            output_r[index] = bilateral_filter_vectorize(buf_r, row, col, width);
            output_g[index] = bilateral_filter_vectorize(buf_g, row, col, width);
            output_b[index] = bilateral_filter_vectorize(buf_b, row, col, width);
        }

        // 可以预取下一行的数据
        const int next_row_prefetch = (row + 1) * width;
        if (next_row_prefetch < base_index + width - 1)
        {
            __builtin_prefetch(&buf_r[next_row_prefetch], 0, 2);
            __builtin_prefetch(&buf_g[next_row_prefetch], 0, 2);
            __builtin_prefetch(&buf_b[next_row_prefetch], 0, 2);
        }
        // if (next_row_prefetch + 1 < base_index + width - 1)
        // {
        //     __builtin_prefetch(&buf_r[next_row_prefetch + 1], 0, 2);
        //     __builtin_prefetch(&buf_g[next_row_prefetch + 1], 0, 2);
        //     __builtin_prefetch(&buf_b[next_row_prefetch + 1], 0, 2);
        // }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    // Save output JPEG image
    const char* output_filepath = argv[2];
    std::cout << "Output file to: " << output_filepath << "\n";
    if (export_jpeg(output_jpeg, output_filepath))
    {
        std::cerr << "Failed to write output JPEG\n";
        return -1;
    }

    // Cleanup
    delete[] output_r_values;
    delete[] output_g_values;
    delete[] output_b_values;
    // print execution time
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
