/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_SCENARIO_CLI_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_SCENARIO_CLI_H

#include "utils/allocate_once.h"
#include "utils/cli_common.h"

#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace nixlbench {

struct ScenarioCommandResult {
    int status = EXIT_SUCCESS;
    bool execute = false;
    std::optional<AllocateOnceRequest> request;
};

bool
isScenarioCommand(int argc, char *argv[]);

bool
supportsAllocateOnce(const PluginMetadata &metadata);

std::vector<std::filesystem::path>
allocateOnceFileNames(const FileOptions &file);

int
parseAllocateOnceCommand(int argc,
                         char *argv[],
                         const std::vector<PluginMetadata> &metadata,
                         AllocateOnceRequest &request,
                         bool &help_requested,
                         std::ostream &out,
                         std::ostream &err);

size_t
allocateOnceWorkingMemory(const AllocateOnceRequest &request);

std::vector<std::string>
allocateOnceBenchmarkArguments(const AllocateOnceRequest &request, const std::string &program_name);

void
printAllocateOncePlan(const AllocateOnceRequest &request, std::ostream &out);

ScenarioCommandResult
prepareScenarioCommand(int argc, char *argv[], std::ostream &out, std::ostream &err);

} // namespace nixlbench

#endif
