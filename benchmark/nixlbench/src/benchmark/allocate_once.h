/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_ALLOCATE_ONCE_H
#define NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_ALLOCATE_ONCE_H

#include "benchmark/scenario.h"
#include "utils/utils.h"

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

/** @brief Fully resolved configuration owned by the allocate-once path. */
struct allocateOnceRequest {
    scenarioConfig common;
    std::vector<std::filesystem::path> files;
    size_t fileSize = 0;
    std::string randomizeLocationMode = XFERBENCH_RANDOMIZE_LOCATION_MODE_BLOCK_ALIGNED;
    uint64_t seed = 0;
    bool managedFiles = true;
    bool direct = false;
};

/** @brief Disjoint block range assigned to one worker thread within one file. */
struct threadFileRegion {
    size_t fileIndex = 0;
    uint64_t firstSlot = 0;
    uint64_t slotCount = 0;
};

std::optional<std::vector<threadFileRegion>>
allocateOnceThreadRegions(const allocateOnceRequest &request, std::string &error);

/** @brief Produces sequential or reproducibly randomized block offsets within one thread region. */
class offsetSequence {
public:
    offsetSequence(threadFileRegion region,
                   size_t batch_size,
                   const std::string &randomize_location_mode,
                   uint64_t seed);

    std::vector<uint64_t>
    next();

private:
    threadFileRegion region_;
    size_t batchSize_;
    bool randomize_;
    uint64_t nextSequentialSlot_ = 0;
    std::mt19937_64 random_;
};

bool
prepareAllocateOnceFiles(const allocateOnceRequest &request, std::ostream &err);

bool
supportsAllocateOnce(const pluginMetadata &metadata);

std::vector<std::filesystem::path>
allocateOnceFileNames(const fileOptions &file);

size_t
allocateOnceWorkingMemory(const allocateOnceRequest &request);

/** @brief Allocate-once CLI, lifecycle policy, and worker-strategy factory. */
class allocateOnceScenario final : public benchmarkScenario {
public:
    allocateOnceScenario();
    ~allocateOnceScenario() override;

    int
    addCommand(CLI::App &scenario,
               const std::vector<pluginMetadata> &metadata,
               std::ostream &err) override;

    bool
    selected() const override;

    int
    finalize(std::ostream &err) override;

    void
    printPlan(std::ostream &out) const override;

    bool
    dryRun() const override;

    bool
    prepare(std::ostream &err) const override;

    legacyWorkerConfig
    legacyWorkerConfiguration() const override;

    std::unique_ptr<xferBenchWorker>
    createWorker(const std::vector<std::string> &devices) const override;

private:
    struct implementation;
    std::unique_ptr<implementation> implementation_;
};

} // namespace nixlbench

#endif // NIXL_BENCHMARK_NIXLBENCH_SRC_BENCHMARK_ALLOCATE_ONCE_H
