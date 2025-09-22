//
// Created by Liu Yuxuan on 2024/9/10
// Email: yuxuanliu1@link.cuhk.edu.cn
//
// Row-wise Pthread parallel implementation of smooth image filtering of JPEG
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <pthread.h>

#include "../utils.hpp"

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg NUM_THREADS\n";
        return -1;
    }
    const char* input_filename = argv[1];
    std::cout << "Input file from: " << input_filename << "\n";
    // Read input JPEG image
    JpegSOA input_jpeg = read_jpeg_soa(input_filename);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    
    /**
     * TODO: Pthread PartC
     */
    // ========== Pthread 双边滤波实现 ==========
    
    // 线程数据结构
    struct ThreadData {
        ColorValue* r_input;
        ColorValue* g_input;
        ColorValue* b_input;
        ColorValue* r_output;
        ColorValue* g_output;
        ColorValue* b_output;
        int width;
        int height;
        int start_row;
        int end_row;
    };
    
    // 线程函数
    auto bilateral_filter_thread = [](void* arg) -> void* {
        ThreadData* data = (ThreadData*)arg;
        
        // 处理分配的行范围（跳过边界行）
        for (int row = data->start_row; row < data->end_row; ++row) {
            for (int col = 1; col < data->width - 1; ++col) {
                int index = row * data->width + col;
                
                // 对三个通道分别应用双边滤波
                data->r_output[index] = bilateral_filter(data->r_input, row, col, data->width);
                data->g_output[index] = bilateral_filter(data->g_input, row, col, data->width);
                data->b_output[index] = bilateral_filter(data->b_input, row, col, data->width);
            }
        }
        return nullptr;
    };
    
    // 获取线程数
    int NUM_THREADS = std::stoi(argv[3]);
    
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
    
    // 创建线程和线程数据
    pthread_t* threads = new pthread_t[NUM_THREADS];
    ThreadData* threadData = new ThreadData[NUM_THREADS];
    
    // 计算每个线程处理的行数
    int rowsPerThread = input_jpeg.height / NUM_THREADS;
    int remainingRows = input_jpeg.height % NUM_THREADS;
    
    // 初始化线程数据
    int currentRow = 0;
    for (int i = 0; i < NUM_THREADS; ++i) {
        int rowsForThisThread = rowsPerThread + (i < remainingRows ? 1 : 0);
        
        // 跳过第一行和最后一行（边界行不处理）
        int start_row = (i == 0) ? 1 : currentRow;
        int end_row = (i == NUM_THREADS - 1) ? input_jpeg.height - 1 : currentRow + rowsForThisThread;
        
        threadData[i] = {
            input_jpeg.r_values,
            input_jpeg.g_values,
            input_jpeg.b_values,
            r_output,
            g_output,
            b_output,
            input_jpeg.width,
            input_jpeg.height,
            start_row,
            end_row
        };
        
        currentRow += rowsForThisThread;
    }
    
    // 创建并启动线程
    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_create(&threads[i], nullptr, bilateral_filter_thread, &threadData[i]);
    }
    
    // 等待所有线程完成
    for (int i = 0; i < NUM_THREADS; ++i) {
        pthread_join(threads[i], nullptr);
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
    delete[] threads;
    delete[] threadData;
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
