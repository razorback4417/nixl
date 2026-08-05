/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_CLI_COMMON_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_CLI_COMMON_H

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include <nixl_types.h>

namespace nixlbench {

struct PluginMetadata {
    std::string name;
    nixl_mem_list_t memory_types;
    nixl_b_params_t parameters;
};

struct FileOptions {
    std::string path;
    std::string filenames;
    int num_files = 1;
    bool direct = false;
};

std::optional<size_t>
parseHumanSize(const std::string &value, std::string &error);

bool
hasMemoryType(const PluginMetadata &metadata, nixl_mem_t memory_type);

bool
validateFileOptions(const FileOptions &file, std::string &error);

std::vector<std::string>
splitFileNames(const std::string &value);

std::string
formatSize(size_t bytes);

std::optional<PluginMetadata>
discoverPluginMetadata(const std::string &name, std::string &error);

std::optional<std::vector<PluginMetadata>>
discoverPluginMetadata(std::string &error);

std::optional<std::vector<PluginMetadata>>
discoverPluginsWithMemoryType(nixl_mem_t memory_type, std::string &error);

std::vector<std::string>
sortedParameterKeys(const nixl_b_params_t &parameters);

std::string
pluginParameterDescription(const nixl_b_params_t &parameters);

} // namespace nixlbench

#endif
