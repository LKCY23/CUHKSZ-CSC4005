import torch
import torch.nn.functional as F
import triton.testing as tt
import triton
import triton.language as tl
import numpy as np
import cv2


@triton.jit
def bilateral_filter_kernel(
    img_pad_ptr,  # *Pointer* to first input vector.
    height,
    width,
    channel,
    pad_size,
    output_ptr,
    stride_h_pad,
    stride_w_pad,
    stride_h_out,
    stride_w_out,
    sigma_density,
    sigma_space,
    ACTIVATION: tl.constexpr,
):

    pid_h = tl.program_id(axis=0)
    pid_w = tl.program_id(axis=1)
    pid_c = tl.program_id(axis=2)

    # 获取中心像素位置（考虑padding）
    center_h = pid_h + pad_size
    center_w = pid_w + pad_size
    
    # 计算中心像素的偏移量
    center_offset = center_h * stride_h_pad + center_w * stride_w_pad + pid_c
    
    # 获取中心像素值
    center_value = tl.load(img_pad_ptr + center_offset)
    
    # 双边滤波计算
    weight_sum = 0.0
    value_sum = 0.0
    
    # 遍历3x3邻域
    for dh in range(-pad_size, pad_size + 1):
        for dw in range(-pad_size, pad_size + 1):
            # 计算邻域像素位置
            neighbor_h = center_h + dh
            neighbor_w = center_w + dw
            neighbor_offset = neighbor_h * stride_h_pad + neighbor_w * stride_w_pad + pid_c
            
            # 加载邻域像素值
            neighbor_value = tl.load(img_pad_ptr + neighbor_offset)
            
            # 计算空间权重（基于距离）
            spatial_dist_sq = dh * dh + dw * dw
            spatial_weight = tl.exp(-spatial_dist_sq / (2.0 * sigma_space * sigma_space))
            
            # 计算强度权重（基于像素值差异）
            intensity_diff = center_value - neighbor_value
            intensity_weight = tl.exp(-intensity_diff * intensity_diff / (2.0 * sigma_density * sigma_density))
            
            # 总权重
            total_weight = spatial_weight * intensity_weight
            
            # 累积加权值
            value_sum += neighbor_value * total_weight
            weight_sum += total_weight
    
    # 计算最终结果
    result = tl.where(weight_sum > 0.0, value_sum / weight_sum, center_value)
    
    # 存储结果
    output_offset = pid_h * stride_h_out + pid_w * stride_w_out + pid_c
    tl.store(output_ptr + output_offset, result)


def bilateral_filter(img_pad, ksize, sigma_space, sigma_density, activation=""):
    """
    双边滤波函数，类似 PartB 的 blur_filter
    """
    assert img_pad.is_contiguous(), "Matrix must be contiguous"
    H, W, C = img_pad.shape
    pad = (ksize - 1) // 2
    H_orig, W_orig = H - 2 * pad, W - 2 * pad  # ignore boundary pixels

    # 创建输出张量
    output = torch.empty(
        (H_orig, W_orig, C), device=img_pad.device, dtype=torch.float32
    )
    
    # 定义网格大小
    grid = lambda META: (
        triton.cdiv(H_orig, 1),
        triton.cdiv(W_orig, 1),
        triton.cdiv(C, 1),
    )

    # 性能测量和执行内核
    elapsed_time = tt.do_bench(
        lambda: bilateral_filter_kernel[grid](
            img_pad,
            H,
            W,
            C,
            pad,
            output,
            img_pad.stride(0),
            img_pad.stride(1),
            output.stride(0),
            output.stride(1),
            sigma_density,
            sigma_space,
            ACTIVATION=activation,
        ),
        warmup=25,  # time to warm up the kernel
        rep=100,
    )
    print(f"Execution Time: {elapsed_time:.2f} ms")
    return output


def main(input_image_path, output_image_path):
    print(f"Input file to: {input_image_path}")

    sigma_space = 1.7
    sigma_density = 50.0
    ksize = 3  # ksize=7

    # read image
    img = cv2.imread(input_image_path, cv2.IMREAD_COLOR).astype(np.float32)

    height, width, channel = img.shape
    # add padding to the image
    pad = (ksize - 1) // 2
    pad_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    pad_img = torch.tensor(pad_img, device="cuda", dtype=torch.float32)

    # 应用双边滤波
    output_triton = bilateral_filter(pad_img, ksize, sigma_space, sigma_density)
    output_img = output_triton.cpu().numpy()

    # 保存输出图像
    print(f"Output file to: {output_image_path}")
    # 将像素数据从 float32 转换为 uint8
    output_img = np.clip(output_img, 0, 255).astype(np.uint8)
    cv2.imwrite(output_image_path, output_img)

    # 清理内存
    del output_triton
    del pad_img
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print(
            "Invalid argument, should be: python3 script.py /path/to/input/jpeg /path/to/output/jpeg"
        )
        sys.exit(-1)
    main(sys.argv[1], sys.argv[2])
