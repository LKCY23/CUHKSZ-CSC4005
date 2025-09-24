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
#include <omp.h>

#include "../utils.hpp"

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg NUM_THREADS\n";
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
    
    
    auto start_time = std::chrono::high_resolution_clock::now();
    // 设置OpenMP线程数
    int NUM_THREADS = std::stoi(argv[3]);
    omp_set_num_threads(NUM_THREADS);
    
    // 分配输出缓冲区
    ColorValue* r_output = new ColorValue[input_jpeg.width * input_jpeg.height];
    ColorValue* g_output = new ColorValue[input_jpeg.width * input_jpeg.height];
    ColorValue* b_output = new ColorValue[input_jpeg.width * input_jpeg.height];
    
    // 复制边界像素到输出缓冲区（优化：只复制边界）
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    
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

    // OpenMP并行双边滤波处理
#pragma omp parallel for shared(input_jpeg, r_output, g_output, b_output) schedule(static)
    for (int row = 1; row < height - 1; ++row) {
        #pragma omp simd
        for (int col = 1; col < width - 1; ++col) {
            int index = row * width + col;
           
            // 对三个通道分别应用双边滤波
            r_output[index] = bilateral_filter(input_jpeg.r_values, row, col, width);
            g_output[index] = bilateral_filter(input_jpeg.g_values, row, col, width);
            b_output[index] = bilateral_filter(input_jpeg.b_values, row, col, width);
        }
    }
    
    // 保存输出JPEG图像
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
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
