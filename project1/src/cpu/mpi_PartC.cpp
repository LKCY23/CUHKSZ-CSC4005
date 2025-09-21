//
// Created by Yang Yufan on 2023/9/16.
// Email: yufanyang1@link.cuhk.edu.cn
//
// MPI implementation of transforming a JPEG image from RGB to gray
//

#include <memory.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <mpi.h> // MPI Header

#include "../utils.hpp"

#define MASTER 0
#define TAG_GATHER 0

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::cerr << "Invalid argument, should be: ./executable "
                     "/path/to/input/jpeg /path/to/output/jpeg\n";
        return -1;
    }
    // Read input JPEG File
    const char* input_filepath = argv[1];
    JpegSOA input_jpeg = read_jpeg_soa(input_filepath);
    if (input_jpeg.r_values == nullptr)
    {
        std::cerr << "Failed to read input JPEG image\n";
        return -1;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    /**
     * TODO: MPI PartC
     */
    // ========== 1) MPI init & 基本信息 ==========
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int height = 0, width = 0;
    if (rank == MASTER) {
        height = input_jpeg.height;
        width  = input_jpeg.width;
    }
    MPI_Bcast(&height, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
    MPI_Bcast(&width,  1, MPI_INT, MASTER, MPI_COMM_WORLD);
    
    // ========== 2) 行划分（尽可能均匀） ==========
    // 所有rank并行计算，避免通信开销
    const int base = height / size;
    const int rem  = height % size;

    std::vector<int> start_row(size), end_row(size), rows_each(size);
    int acc = 0;
    for (int r = 0; r < size; ++r) {
        int rows = base + (r < rem ? 1 : 0);   // 该 rank 负责的“内部行数”（不含 halo）
        start_row[r] = acc;
        end_row[r]   = acc + rows;
        rows_each[r] = rows;
        acc += rows;
    }

    // rank 的内部行范围（全局行号，半开区间）
    const int my_start = start_row[rank];
    const int my_end   = end_row[rank];
    const int my_rows  = rows_each[rank];  // 可能为 0（极端情况）

    // ========== 3) 计算散发时要附带的 halo（只上下各 1 行） ==========
    // 只有当该 rank 实际有内部行时才需要 halo
    const int top_halo = (rank > 0     && my_rows > 0) ? 1 : 0;
    const int bot_halo = (rank < size-1 && my_rows > 0) ? 1 : 0;

    // 本地缓冲覆盖的全局行范围（含 halo）
    const int local_start = my_start - top_halo;
    const int local_end   = my_end   + bot_halo;
    const int local_h     = (my_rows > 0) ? (local_end - local_start) : 0;
    const int stride      = width;

    // 为 SoA 三个通道分配本地缓冲（含 halo）
    std::vector<ColorValue> r_in(local_h * stride), g_in(local_h * stride), b_in(local_h * stride);
    std::vector<ColorValue> r_out(local_h * stride), g_out(local_h * stride), b_out(local_h * stride);
    
    // 使用JpegSOA结构
    // JpegSOA local_input, local_output;
    // local_input.height = local_h;
    // local_input.width = width;
    // local_input.r_values = new ColorValue[local_h * stride];
    // local_input.g_values = new ColorValue[local_h * stride];
    // local_input.b_values = new ColorValue[local_h * stride];
    
    // local_output.height = local_h;
    // local_output.width = width;
    // local_output.r_values = new ColorValue[local_h * stride];
    // local_output.g_values = new ColorValue[local_h * stride];
    // local_output.b_values = new ColorValue[local_h * stride];

    // ========== 4) MASTER 一次性分发（内部行 + 必要的上下 halo） ==========
    // Scatterv 的计数/位移（单位：元素个数）
    std::vector<int> scounts(size), sdispls(size);
    if (rank == MASTER) {
        for (int r = 0; r < size; ++r) {
            int rr = rows_each[r];
            int th = (r > 0      && rr > 0) ? 1 : 0;
            int bh = (r < size-1 && rr > 0) ? 1 : 0;
            int lstart = start_row[r] - th;
            int lend   = end_row[r]   + bh;
            int lh     = (rr > 0) ? (lend - lstart) : 0;

            sdispls[r] = lstart * width;
            scounts[r] = lh * width;
        }
    }

    // 三个通道分别 Scatterv。根的 sendbuf 指向 input_jpeg.*；各 rank recv 到各自 r_in/g_in/b_in
    MPI_Scatterv(input_jpeg.r_values,
                 (rank == MASTER ? scounts.data() : nullptr),
                 (rank == MASTER ? sdispls.data() : nullptr),
                 MPI_UNSIGNED_CHAR,
                 (local_h > 0 ? r_in.data() : nullptr),
                 local_h * width, MPI_UNSIGNED_CHAR,
                 MASTER, MPI_COMM_WORLD);

    MPI_Scatterv(input_jpeg.g_values,
                 (rank == MASTER ? scounts.data() : nullptr),
                 (rank == MASTER ? sdispls.data() : nullptr),
                 MPI_UNSIGNED_CHAR,
                 (local_h > 0 ? g_in.data() : nullptr),
                 local_h * width, MPI_UNSIGNED_CHAR,
                 MASTER, MPI_COMM_WORLD);

    MPI_Scatterv(input_jpeg.b_values,
                 (rank == MASTER ? scounts.data() : nullptr),
                 (rank == MASTER ? sdispls.data() : nullptr),
                 MPI_UNSIGNED_CHAR,
                 (local_h > 0 ? b_in.data() : nullptr),
                 local_h * width, MPI_UNSIGNED_CHAR,
                 MASTER, MPI_COMM_WORLD);

    // 使用JpegSOA结构的Scatterv
    // MPI_Scatterv(input_jpeg.r_values,
    //              (rank == MASTER ? scounts.data() : nullptr),
    //              (rank == MASTER ? sdispls.data() : nullptr),
    //              MPI_UNSIGNED_CHAR,
    //              (local_h > 0 ? local_input.r_values : nullptr),
    //              local_h * width, MPI_UNSIGNED_CHAR,
    //              MASTER, MPI_COMM_WORLD);

    // MPI_Scatterv(input_jpeg.g_values,
    //              (rank == MASTER ? scounts.data() : nullptr),
    //              (rank == MASTER ? sdispls.data() : nullptr),
    //              MPI_UNSIGNED_CHAR,
    //              (local_h > 0 ? local_input.g_values : nullptr),
    //              local_h * width, MPI_UNSIGNED_CHAR,
    //              MASTER, MPI_COMM_WORLD);

    // MPI_Scatterv(input_jpeg.b_values,
    //              (rank == MASTER ? scounts.data() : nullptr),
    //              (rank == MASTER ? sdispls.data() : nullptr),
    //              MPI_UNSIGNED_CHAR,
    //              (local_h > 0 ? local_input.b_values : nullptr),
    //              local_h * width, MPI_UNSIGNED_CHAR,
    //              MASTER, MPI_COMM_WORLD);

    // ========== 5) 本地计算：只处理全局 [1..H-2] × [1..W-2] ==========
    if (local_h > 0) {
        // 该 rank 实际要计算的全局行（去掉全局边界 0 和 H-1）
        int proc_g_start = (my_start < 1) ? 1 : my_start;
        int proc_g_end   = (my_end   > height - 1) ? (height - 1) : my_end; // 半开区间，最多到 H-1

        // 转成本地行号（相对于 local_start）
        int lstart = proc_g_start - local_start;
        int lend   = proc_g_end   - local_start;

        for (int lr = lstart; lr < lend; ++lr) {
            // 列同样跳过全局边界：1 .. W-2
            for (int c = 1; c < width - 1; ++c) {
                int idx = lr * stride + c;
                r_out[idx] = bilateral_filter(r_in.data(), lr, c, width);
                g_out[idx] = bilateral_filter(g_in.data(), lr, c, width);
                b_out[idx] = bilateral_filter(b_in.data(), lr, c, width);
                
                // 使用JpegSOA结构
                // local_output.r_values[idx] = bilateral_filter(local_input.r_values, lr, c, width);
                // local_output.g_values[idx] = bilateral_filter(local_input.g_values, lr, c, width);
                // local_output.b_values[idx] = bilateral_filter(local_input.b_values, lr, c, width);
            }
        }
    }

    // ========== 6) 仅把“已计算的内部行”收回到 MASTER ==========
    // 每个 rank 回传的“内部行数”（不含全局第 0 行/第 H-1 行）
    int send_rows = my_rows;
    if (rank == 0      && send_rows > 0) --send_rows;            // 去掉全局第 0 行
    if (rank == size-1 && send_rows > 0) --send_rows;            // 去掉全局第 H-1 行（即最后一行）

    // 本地发送指针：从“第一条已计算行”的本地行号开始
    int proc_g_start = (my_start < 1) ? 1 : my_start;             // 已计算的全局起始行
    int lfirst       = (local_h > 0) ? (proc_g_start - local_start) : 0;

    ColorValue* r_send = (send_rows > 0) ? &r_out[lfirst * stride] : nullptr;
    ColorValue* g_send = (send_rows > 0) ? &g_out[lfirst * stride] : nullptr;
    ColorValue* b_send = (send_rows > 0) ? &b_out[lfirst * stride] : nullptr;
    
    // 使用JpegSOA结构
    // ColorValue* r_send = (send_rows > 0) ? &local_output.r_values[lfirst * stride] : nullptr;
    // ColorValue* g_send = (send_rows > 0) ? &local_output.g_values[lfirst * stride] : nullptr;
    // ColorValue* b_send = (send_rows > 0) ? &local_output.b_values[lfirst * stride] : nullptr;

    // MASTER 端的接收 counts / displs（单位：元素个数）
    std::vector<int> rcounts, rdispls;
    if (rank == MASTER) {
        rcounts.resize(size);
        rdispls.resize(size);
        for (int r = 0; r < size; ++r) {
            int rows = rows_each[r];
            if (r == 0      && rows > 0) --rows;                 // 去掉全局第 0 行
            if (r == size-1 && rows > 0) --rows;                 // 去掉全局第 H-1 行
            // if (rows < 0) rows = 0;
            rcounts[r] = rows * width;

            // 这段数据在整图中的起始偏移：从该 rank 的第一条“已计算行”开始
            int first_row_global = start_row[r] + ((r == 0) ? 1 : 0);
            rdispls[r] = first_row_global * width;
        }
    }

    MPI_Gatherv(r_send, send_rows * width, MPI_UNSIGNED_CHAR,
                input_jpeg.r_values,
                (rank == MASTER ? rcounts.data() : nullptr),
                (rank == MASTER ? rdispls.data() : nullptr),
                MPI_UNSIGNED_CHAR, MASTER, MPI_COMM_WORLD);

    MPI_Gatherv(g_send, send_rows * width, MPI_UNSIGNED_CHAR,
                input_jpeg.g_values,
                (rank == MASTER ? rcounts.data() : nullptr),
                (rank == MASTER ? rdispls.data() : nullptr),
                MPI_UNSIGNED_CHAR, MASTER, MPI_COMM_WORLD);

    MPI_Gatherv(b_send, send_rows * width, MPI_UNSIGNED_CHAR,
                input_jpeg.b_values,
                (rank == MASTER ? rcounts.data() : nullptr),
                (rank == MASTER ? rdispls.data() : nullptr),
                MPI_UNSIGNED_CHAR, MASTER, MPI_COMM_WORLD);

    // 清理JpegSOA结构的内存
    // delete[] local_input.r_values;
    // delete[] local_input.g_values;
    // delete[] local_input.b_values;
    // delete[] local_output.r_values;
    // delete[] local_output.g_values;
    // delete[] local_output.b_values;
    
    MPI_Finalize();
    
    // Save output JPEG image (只有Master进程保存)
    if (rank == MASTER) {
        const char* output_filepath = argv[2];
        std::cout << "Output file to: " << output_filepath << "\n";
        if (export_jpeg(input_jpeg, output_filepath))
        {
            std::cerr << "Failed to write output JPEG\n";
            return -1;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);
    std::cout << "Transformation Complete!" << std::endl;
    std::cout << "Execution Time: " << elapsed_time.count()
              << " milliseconds\n";
    return 0;
}
