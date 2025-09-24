//
// Created by Zhong Yebin on 2023/9/16.
// Email: yebinzhong@link.cuhk.edu.cn
//
// OpenACC implementation of image filtering on JPEG
//

#include <memory.h>
#include <cstring>
#include <chrono>
#include <cmath>
#include <iostream>
#include <openacc.h>

#include "../utils.hpp"

const float spatial_filter[FILTERSIZE][FILTERSIZE] = { // [Bilateral Filter Only] spatial weights
    {expf(-2.0f / (2.0f * SIGMA_D * SIGMA_D)), expf(-1.0f / (2.0f * SIGMA_D * SIGMA_D)), expf(-2.0f / (2.0f * SIGMA_D * SIGMA_D))},
    {expf(-1.0f / (2.0f * SIGMA_D * SIGMA_D)), expf(-0.0f / (2.0f * SIGMA_D * SIGMA_D)), expf(-1.0f / (2.0f * SIGMA_D * SIGMA_D))},
    {expf(-2.0f / (2.0f * SIGMA_D * SIGMA_D)), expf(-1.0f / (2.0f * SIGMA_D * SIGMA_D)), expf(-2.0f / (2.0f * SIGMA_D * SIGMA_D))}};
    
#pragma acc routine seq
ColorValue acc_clamp_pixel_value(float value)
{
    return value > 255 ? 255
           : value < 0 ? 0
                       : static_cast<unsigned char>(value);
}

#pragma acc routine seq
ColorValue acc_bilateral_filter(const ColorValue* values, int pixel_id, int width,
                               const float (&spatial_filter)[FILTERSIZE][FILTERSIZE])
{
    float weight_sum = 0.0f;
    float value_sum = 0.0f;
    ColorValue center_value = values[pixel_id];
    
    // 像PartB一样简洁的循环展开写法，只计算强度权重
    // 中心点 (1,1)
    float intensity_weight = expf(-(0.0f) / (2.0f * SIGMA_R * SIGMA_R));
    value_sum += values[pixel_id] * spatial_filter[1][1] * intensity_weight;
    weight_sum += spatial_filter[1][1] * intensity_weight;
    
    // 左 (1,0)
    intensity_weight = expf(-((center_value - values[pixel_id - 1]) * (center_value - values[pixel_id - 1])) / (2.0f * SIGMA_R * SIGMA_R));
    value_sum += values[pixel_id - 1] * spatial_filter[1][0] * intensity_weight;
    weight_sum += spatial_filter[1][0] * intensity_weight;
    
    // 右 (1,2)
    intensity_weight = expf(-((center_value - values[pixel_id + 1]) * (center_value - values[pixel_id + 1])) / (2.0f * SIGMA_R * SIGMA_R));
    value_sum += values[pixel_id + 1] * spatial_filter[1][2] * intensity_weight;
    weight_sum += spatial_filter[1][2] * intensity_weight;
    
    // 上 (0,1)
    intensity_weight = expf(-((center_value - values[pixel_id - width]) * (center_value - values[pixel_id - width])) / (2.0f * SIGMA_R * SIGMA_R));
    value_sum += values[pixel_id - width] * spatial_filter[0][1] * intensity_weight;
    weight_sum += spatial_filter[0][1] * intensity_weight;
    
    // 左上 (0,0)
    intensity_weight = expf(-((center_value - values[pixel_id - width - 1]) * (center_value - values[pixel_id - width - 1])) / (2.0f * SIGMA_R * SIGMA_R));
    value_sum += values[pixel_id - width - 1] * spatial_filter[0][0] * intensity_weight;
    weight_sum += spatial_filter[0][0] * intensity_weight;
    
    // 右上 (0,2)
    intensity_weight = expf(-((center_value - values[pixel_id - width + 1]) * (center_value - values[pixel_id - width + 1])) / (2.0f * SIGMA_R * SIGMA_R));
    value_sum += values[pixel_id - width + 1] * spatial_filter[0][2] * intensity_weight;
    weight_sum += spatial_filter[0][2] * intensity_weight;
    
    // 下 (2,1)
    intensity_weight = expf(-((center_value - values[pixel_id + width]) * (center_value - values[pixel_id + width])) / (2.0f * SIGMA_R * SIGMA_R));
    value_sum += values[pixel_id + width] * spatial_filter[2][1] * intensity_weight;
    weight_sum += spatial_filter[2][1] * intensity_weight;
    
    // 左下 (2,0)
    intensity_weight = expf(-((center_value - values[pixel_id + width - 1]) * (center_value - values[pixel_id + width - 1])) / (2.0f * SIGMA_R * SIGMA_R));
    value_sum += values[pixel_id + width - 1] * spatial_filter[2][0] * intensity_weight;
    weight_sum += spatial_filter[2][0] * intensity_weight;
    
    // 右下 (2,2)
    intensity_weight = expf(-((center_value - values[pixel_id + width + 1]) * (center_value - values[pixel_id + width + 1])) / (2.0f * SIGMA_R * SIGMA_R));
    value_sum += values[pixel_id + width + 1] * spatial_filter[2][2] * intensity_weight;
    weight_sum += spatial_filter[2][2] * intensity_weight;
    
    float result = (weight_sum > 0) ? (value_sum / weight_sum) : center_value;
    return acc_clamp_pixel_value(result);
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
    /**
     * TODO: OpenACC PartC
     */
    
    // ========== OpenACC 双边滤波实现 ==========
    
    // 获取图像尺寸
    int width = input_jpeg.width;
    int height = input_jpeg.height;
    size_t buffer_size = width * height;
    
    // 分配输出缓冲区
    ColorValue* r_output = new ColorValue[buffer_size];
    ColorValue* g_output = new ColorValue[buffer_size];
    ColorValue* b_output = new ColorValue[buffer_size];
    
    // 初始化输出缓冲区（只复制边界像素，内部区域保持未初始化）
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
    
    // 创建指向数据的指针，避免结构体成员访问问题
    ColorValue* r_values = input_jpeg.r_values;
    ColorValue* g_values = input_jpeg.g_values;
    ColorValue* b_values = input_jpeg.b_values;
    
    // 将数据复制到GPU
#pragma acc enter data copyin(r_values[0:buffer_size], \
                              g_values[0:buffer_size], \
                              b_values[0:buffer_size], \
                              r_output[0:buffer_size], \
                              g_output[0:buffer_size], \
                              b_output[0:buffer_size], \
                              spatial_filter[0:FILTERSIZE][0:FILTERSIZE])
    
    // 更新GPU上的输出数据（边界像素）
#pragma acc update device(r_output[0:buffer_size], \
                          g_output[0:buffer_size], \
                          b_output[0:buffer_size], \
                          spatial_filter[0:FILTERSIZE][0:FILTERSIZE])

    auto start_time = std::chrono::high_resolution_clock::now();
    // GPU并行计算双边滤波
#pragma acc parallel present(r_values[0:buffer_size], \
                            g_values[0:buffer_size], \
                            b_values[0:buffer_size], \
                            r_output[0:buffer_size], \
                            g_output[0:buffer_size], \
                            b_output[0:buffer_size], \
                            spatial_filter[0:FILTERSIZE][0:FILTERSIZE]) num_gangs(1024) vector_length(256) 
    {
#pragma acc loop independent collapse(2)
        for (int row = 1; row < height - 1; ++row) {
            for (int col = 1; col < width - 1; ++col) {
                int index = row * width + col;
                
                // 对三个通道分别应用双边滤波
                r_output[index] = acc_bilateral_filter(r_values, index, width, spatial_filter);
                g_output[index] = acc_bilateral_filter(g_values, index, width, spatial_filter);
                b_output[index] = acc_bilateral_filter(b_values, index, width, spatial_filter);
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    
    // 将结果从GPU复制回CPU
#pragma acc update self(r_output[0:buffer_size], \
                        g_output[0:buffer_size], \
                        b_output[0:buffer_size])
    
    // 清理GPU内存
#pragma acc exit data copyout(r_output[0:buffer_size], \
                              g_output[0:buffer_size], \
                              b_output[0:buffer_size])
    
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
    
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    
    return 0;
}
