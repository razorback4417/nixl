/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_SCENARIO_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_SCENARIO_H

#include "utils/cli_common.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace CLI {
class App;
}

class xferBenchWorker;

namespace nixlbench {

/** @brief CLI values shared by every modeled scenario. */
struct scenarioOptions {
    std::string blockSize;
    size_t batchSize = 1;
    int threads = 1;
    int iterations = 1000;
    int warmupIterations = 10;
    std::string operation = "write";
    std::string initiatorMemory = "auto";
    bool checkConsistency = false;
    bool dryRun = false;
};

/** @brief Validated common configuration consumed by scenario worker strategies. */
struct scenarioConfig {
    std::string pluginName;
    nixl_b_params_t pluginParameters;
    size_t blockSize = 0;
    size_t batchSize = 1;
    int threads = 1;
    int iterations = 1000;
    int warmupIterations = 10;
    nixl_xfer_op_t operation = NIXL_WRITE;
    nixl_mem_t initiatorMemory = DRAM_SEG;
    bool checkConsistency = false;
    bool dryRun = false;
};

/** @brief Typed input to the legacy xferBenchConfig compatibility adapter. */
struct legacyWorkerConfig {
    scenarioConfig common;
    size_t workingMemory = 0;
    nixl_mem_t targetMemory = DRAM_SEG;
    bool recreateTransferRequest = false;
    std::vector<std::string> fileNames;
    bool storageDirect = false;
    std::string randomizeLocationMode;
    uint64_t randomizeLocationSeed = 0;
};

struct scenarioPluginBinding {
    pluginMetadata metadata;
    CLI::App *command = nullptr;
    std::vector<std::pair<std::string, std::string>> overrides;
};

using scenario_plugin_filter_t = std::function<bool(const pluginMetadata &)>;
using scenario_plugin_bindings_t = std::vector<std::unique_ptr<scenarioPluginBinding>>;

void
addCommonScenarioOptions(CLI::App &command, scenarioOptions &options);

bool
resolveCommonScenarioOptions(const scenarioOptions &options,
                             const pluginMetadata &metadata,
                             const std::vector<std::pair<std::string, std::string>> &overrides,
                             scenarioConfig &config,
                             std::ostream &err);

int
addScenarioPluginCommands(CLI::App &scenario,
                          const std::vector<pluginMetadata> &metadata,
                          const scenario_plugin_filter_t &filter,
                          scenario_plugin_bindings_t &bindings,
                          std::ostream &err);

const scenarioPluginBinding *
selectedScenarioPlugin(const scenario_plugin_bindings_t &bindings);

void
addFileScenarioOptions(CLI::App &command, fileOptions &options);

std::vector<std::string>
legacyWorkerArguments(const legacyWorkerConfig &config, const std::string &program_name);

/**
 * @brief Extension point for a complete benchmark path.
 *
 * A scenario owns its specific CLI, validation, resource preparation, and worker strategy. The
 * command dispatcher and shared worker path depend only on this interface.
 */
class benchmarkScenario {
public:
    virtual ~benchmarkScenario() = default;

    virtual int
    addCommand(CLI::App &scenario,
               const std::vector<pluginMetadata> &metadata,
               std::ostream &err) = 0;

    virtual bool
    selected() const = 0;

    virtual int
    finalize(std::ostream &err) = 0;

    virtual void
    printPlan(std::ostream &out) const = 0;

    virtual bool
    dryRun() const = 0;

    virtual bool
    prepare(std::ostream &err) const = 0;

    virtual legacyWorkerConfig
    legacyWorkerConfiguration() const = 0;

    virtual std::unique_ptr<xferBenchWorker>
    createWorker(const std::vector<std::string> &devices) const = 0;
};

struct scenarioCommandResult {
    int status = EXIT_SUCCESS;
    bool execute = false;
    std::unique_ptr<benchmarkScenario> scenario;
};

bool
isScenarioCommand(int argc, char *argv[]);

scenarioCommandResult
prepareScenarioCommand(int argc, char *argv[], std::ostream &out, std::ostream &err);

scenarioCommandResult
prepareScenarioCommand(int argc,
                       char *argv[],
                       const std::vector<pluginMetadata> &metadata,
                       std::ostream &out,
                       std::ostream &err);

} // namespace nixlbench

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_SCENARIO_H
