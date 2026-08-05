/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_RAW_CLI_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_UTILS_RAW_CLI_H

#include "utils/cli_common.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

#include <nixl_types.h>

namespace nixlbench {

struct rawOptions {
    std::string operation = "WRITE";
    size_t totalBufferSize = 8ULL * 1024 * 1024 * 1024;
    size_t startBlockSize = 4ULL * 1024;
    size_t maxBlockSize = 64ULL * 1024 * 1024;
    size_t startBatchSize = 1;
    size_t maxBatchSize = 1;
    int iterations = 1000;
    int warmupIterations = 100;
    int threads = 1;
    int pipelineDepth = 1;
    bool checkConsistency = false;
    bool dryRun = false;
};

struct rawPosixRequest {
    rawOptions raw;
    fileOptions file;
    bool hasFileOptions = false;
    nixl_b_params_t pluginParameters;
};

struct rawCommandResult {
    int status = EXIT_SUCCESS;
    bool execute = false;
    std::optional<nixl_b_params_t> pluginParameters;
};

bool
isRawCommand(int argc, char *argv[]);

int
parseRawPosixCommand(int argc,
                     char *argv[],
                     const pluginMetadata &metadata,
                     rawPosixRequest &request,
                     bool &help_requested,
                     std::ostream &out,
                     std::ostream &err);

int
parseRawCommand(int argc,
                char *argv[],
                const std::vector<pluginMetadata> &file_plugins,
                rawPosixRequest &request,
                bool &help_requested,
                std::ostream &out,
                std::ostream &err);

std::vector<std::string>
benchmarkFileArguments(const rawPosixRequest &request, const std::string &program_name);

void
printRawPosixPlan(const rawPosixRequest &request,
                  const pluginMetadata &metadata,
                  int normalized_iterations,
                  int normalized_warmup_iterations,
                  std::ostream &out);

rawCommandResult
prepareRawCommand(int argc, char *argv[], std::ostream &out, std::ostream &err);

} // namespace nixlbench

#endif
