/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_ALLOCATE_ONCE_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_ALLOCATE_ONCE_H

#include <nixl_types.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace nixlbench {

enum class OffsetMode {
    Random,
    Sequential,
};

struct AllocateOnceRequest {
    std::string plugin_name;
    nixl_b_params_t plugin_parameters;
    std::vector<std::filesystem::path> files;
    size_t file_size = 0;
    size_t block_size = 0;
    size_t batch_size = 1;
    int threads = 1;
    int iterations = 1000;
    int warmup_iterations = 10;
    nixl_xfer_op_t operation = NIXL_WRITE;
    OffsetMode offset_mode = OffsetMode::Random;
    uint64_t seed = 0;
    nixl_mem_t initiator_memory = DRAM_SEG;
    bool managed_files = true;
    bool direct = false;
    bool check_consistency = false;
    bool dry_run = false;
};

struct ThreadFileRegion {
    size_t file_index = 0;
    uint64_t first_slot = 0;
    uint64_t slot_count = 0;
};

std::optional<std::vector<ThreadFileRegion>>
allocateOnceThreadRegions(const AllocateOnceRequest &request, std::string &error);

class OffsetSequence {
public:
    OffsetSequence(ThreadFileRegion region, size_t batch_size, OffsetMode mode, uint64_t seed);

    std::vector<uint64_t>
    next();

private:
    ThreadFileRegion region_;
    size_t batch_size_;
    OffsetMode mode_;
    uint64_t next_sequential_slot_ = 0;
    std::mt19937_64 random_;
};

bool
prepareAllocateOnceFiles(const AllocateOnceRequest &request, std::ostream &err);

} // namespace nixlbench

#endif
