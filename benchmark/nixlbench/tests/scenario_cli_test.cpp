/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark/scenario.h"
#include "benchmark/allocate_once.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

namespace nixlbench {
namespace {

    class testArguments {
    public:
        testArguments(std::initializer_list<const char *> values) {
            for (const auto *value : values) {
                storage_.emplace_back(value);
            }
            for (auto &value : storage_) {
                pointers_.push_back(value.data());
            }
        }

        int
        argc() const {
            return static_cast<int>(pointers_.size());
        }

        char **
        argv() {
            return pointers_.data();
        }

    private:
        std::vector<std::string> storage_;
        std::vector<char *> pointers_;
    };

    pluginMetadata
    posixMetadata() {
        return {"POSIX",
                {DRAM_SEG, FILE_SEG},
                {{"future_parameter", "default"}, {"ios_pool_size", "4096"}}};
    }

    pluginMetadata
    futureFileMetadata() {
        return {
            "FUTURE-FILE", {VRAM_SEG, DRAM_SEG, FILE_SEG}, {{"provider-key", "provider default"}}};
    }

    scenarioCommandResult
    parse(testArguments &arguments,
          const std::vector<pluginMetadata> &metadata,
          std::ostringstream &out,
          std::ostringstream &err) {
        return prepareScenarioCommand(arguments.argc(), arguments.argv(), metadata, out, err);
    }

    TEST(ScenarioCliDispatchTest, OnlyExplicitScenarioSelectsTheScenarioParser) {
        testArguments scenario{"nixlbench", "scenario", "allocate-once"};
        EXPECT_TRUE(isScenarioCommand(scenario.argc(), scenario.argv()));

        testArguments raw{"nixlbench", "raw", "posix"};
        EXPECT_FALSE(isScenarioCommand(raw.argc(), raw.argv()));

        testArguments legacy{"nixlbench", "--backend=POSIX"};
        EXPECT_FALSE(isScenarioCommand(legacy.argc(), legacy.argv()));
    }

    TEST(ScenarioMetadataTest, CompatibilityComesFromAdvertisedMemoryTypes) {
        EXPECT_TRUE(supportsAllocateOnce(posixMetadata()));
        EXPECT_TRUE(supportsAllocateOnce(futureFileMetadata()));
        EXPECT_FALSE(supportsAllocateOnce({"UCX", {DRAM_SEG}, {}}));
        EXPECT_FALSE(supportsAllocateOnce({"FILE_ONLY", {FILE_SEG}, {}}));
    }

    TEST(ScenarioParserTest, AcceptsCompatibleFuturePluginAndPreservesOpaqueParameters) {
        testArguments arguments{"nixlbench",
                                "scenario",
                                "allocate-once",
                                "--file-size",
                                "1MiB",
                                "--block-size",
                                "4KiB",
                                "future-file",
                                "--plugin-param",
                                "provider-key",
                                "Exact Value"};
        std::ostringstream out;
        std::ostringstream err;

        auto result = parse(arguments, {futureFileMetadata()}, out, err);
        ASSERT_EQ(result.status, 0) << err.str();
        ASSERT_NE(result.scenario, nullptr);
        const auto config = result.scenario->legacyWorkerConfiguration();
        EXPECT_EQ(config.common.pluginParameters.at("provider-key"), "Exact Value");
        const auto benchmark_arguments = legacyWorkerArguments(config, "nixlbench");
        EXPECT_NE(std::find(benchmark_arguments.begin(),
                            benchmark_arguments.end(),
                            "--backend=FUTURE-FILE"),
                  benchmark_arguments.end());
        EXPECT_NE(std::find(benchmark_arguments.begin(),
                            benchmark_arguments.end(),
                            "--initiator_seg_type=VRAM"),
                  benchmark_arguments.end());
    }

    TEST(ScenarioParserTest, ScenarioOptionsWorkBeforeOrAfterPluginSelection) {
        testArguments before{"nixlbench",
                             "scenario",
                             "allocate-once",
                             "--file-size",
                             "1MiB",
                             "--block-size",
                             "4KiB",
                             "--threads",
                             "2",
                             "posix",
                             "--num-files",
                             "2"};
        testArguments after{"nixlbench",
                            "scenario",
                            "allocate-once",
                            "posix",
                            "--file-size",
                            "1MiB",
                            "--block-size",
                            "4KiB",
                            "--threads",
                            "2",
                            "--num-files",
                            "2"};
        for (auto *arguments : {&before, &after}) {
            std::ostringstream out;
            std::ostringstream err;
            auto result = parse(*arguments, {posixMetadata()}, out, err);
            ASSERT_EQ(result.status, 0) << err.str();
            EXPECT_NE(out.str().find("worker threads: 2"), std::string::npos);
            EXPECT_NE(out.str().find("files: 2"), std::string::npos);
        }
    }

    TEST(ScenarioParserTest, RejectsInvalidCapacityOwnershipAndMemoryRequests) {
        const auto expect_failure = [](testArguments &arguments, const std::string &message) {
            std::ostringstream out;
            std::ostringstream err;
            EXPECT_NE(parse(arguments, {posixMetadata()}, out, err).status, 0);
            EXPECT_NE(err.str().find(message), std::string::npos) << err.str();
        };

        testArguments too_small{"nixlbench",
                                "scenario",
                                "allocate-once",
                                "--file-size",
                                "4KiB",
                                "--block-size",
                                "4KiB",
                                "--batch-size",
                                "2",
                                "posix"};
        expect_failure(too_small, "--batch-size");

        testArguments fractional_block{"nixlbench",
                                       "scenario",
                                       "allocate-once",
                                       "--file-size",
                                       "10KiB",
                                       "--block-size",
                                       "4KiB",
                                       "posix"};
        expect_failure(fractional_block, "exact multiple");

        testArguments mixed_ownership{"nixlbench",
                                      "scenario",
                                      "allocate-once",
                                      "--file-size",
                                      "1MiB",
                                      "--block-size",
                                      "4KiB",
                                      "posix",
                                      "--path",
                                      "/tmp",
                                      "--filenames",
                                      "/tmp/a"};
        expect_failure(mixed_ownership, "mutually exclusive");

        testArguments unsupported_memory{"nixlbench",
                                         "scenario",
                                         "allocate-once",
                                         "--file-size",
                                         "1MiB",
                                         "--block-size",
                                         "4KiB",
                                         "--initiator-memory",
                                         "vram",
                                         "posix"};
        expect_failure(unsupported_memory, "does not advertise VRAM_SEG");

        testArguments sequential_with_seed{"nixlbench",
                                           "scenario",
                                           "allocate-once",
                                           "--file-size",
                                           "1MiB",
                                           "--block-size",
                                           "4KiB",
                                           "--randomize-location-mode",
                                           "none",
                                           "--seed",
                                           "7",
                                           "posix"};
        expect_failure(sequential_with_seed, "--seed requires");
    }

    TEST(ScenarioParserTest, ExplicitMemoryDoesNotSilentlyFallBack) {
        testArguments arguments{"nixlbench",
                                "scenario",
                                "allocate-once",
                                "--file-size",
                                "1MiB",
                                "--block-size",
                                "4KiB",
                                "--initiator-memory",
                                "dram",
                                "future-file"};
        std::ostringstream out;
        std::ostringstream err;

        auto result = parse(arguments, {futureFileMetadata()}, out, err);
        ASSERT_EQ(result.status, 0) << err.str();
        EXPECT_NE(out.str().find("initiator memory: DRAM"), std::string::npos);
    }

    TEST(ScenarioPlanTest, SeparatesDatasetSizeFromBoundedTransferMemory) {
        allocateOnceRequest request;
        request.common.pluginName = "POSIX";
        request.common.pluginParameters = {{"ios_pool_size", "4096"}};
        request.fileSize = 1024 * 1024;
        request.common.blockSize = 4096;
        request.common.batchSize = 4;
        request.common.threads = 3;
        request.files = {"/tmp/nixlbench_allocate_once_0.dat"};
        request.common.initiatorMemory = DRAM_SEG;
        request.common.dryRun = true;

        EXPECT_EQ(allocateOnceWorkingMemory(request), 3U * 4U * 4096U);

        testArguments arguments{"nixlbench",
                                "scenario",
                                "allocate-once",
                                "--file-size",
                                "1MiB",
                                "--block-size",
                                "4KiB",
                                "--batch-size",
                                "4",
                                "--threads",
                                "3",
                                "--dry-run",
                                "posix"};
        std::ostringstream out;
        std::ostringstream err;
        const auto result = parse(arguments, {posixMetadata()}, out, err);
        ASSERT_EQ(result.status, 0) << err.str();
        EXPECT_NE(out.str().find("size per file: 1 MiB"), std::string::npos);
        EXPECT_NE(out.str().find("working memory: 48 KiB"), std::string::npos);
        EXPECT_NE(out.str().find("open, allocate, and register once"), std::string::npos);
        EXPECT_NE(out.str().find("no backing file was opened"), std::string::npos);
    }

    TEST(ScenarioPlanTest, MapsWorkloadIntentIntoTheSharedWorkerConfiguration) {
        testArguments command{"nixlbench",
                              "scenario",
                              "allocate-once",
                              "--file-size",
                              "1MiB",
                              "--block-size",
                              "4KiB",
                              "--batch-size",
                              "4",
                              "--threads",
                              "2",
                              "--iterations",
                              "7",
                              "--warmup-iterations",
                              "3",
                              "future-file",
                              "--num-files",
                              "2"};
        std::ostringstream out;
        std::ostringstream err;
        auto result = parse(command, {futureFileMetadata()}, out, err);
        ASSERT_EQ(result.status, 0) << err.str();
        const auto config = result.scenario->legacyWorkerConfiguration();
        EXPECT_EQ(config.common.pluginName, "FUTURE-FILE");
        EXPECT_EQ(config.common.initiatorMemory, VRAM_SEG);
        EXPECT_EQ(config.targetMemory, FILE_SEG);
        EXPECT_EQ(config.workingMemory, 32768U);
        EXPECT_TRUE(config.recreateTransferRequest);
        EXPECT_EQ(config.fileNames.size(), 2U);
        EXPECT_EQ(config.randomizeLocationMode, XFERBENCH_RANDOMIZE_LOCATION_MODE_BLOCK_ALIGNED);

        const auto arguments = legacyWorkerArguments(config, "nixlbench");
        const auto contains = [&](const std::string &argument) {
            return std::find(arguments.begin(), arguments.end(), argument) != arguments.end();
        };
        EXPECT_TRUE(contains("--backend=FUTURE-FILE"));
        EXPECT_TRUE(contains("--initiator_seg_type=VRAM"));
        EXPECT_TRUE(contains("--target_seg_type=FILE"));
        EXPECT_TRUE(contains("--total_buffer_size=32768"));
        EXPECT_TRUE(contains("--num_iter=14"));
        EXPECT_TRUE(contains("--warmup_iter=6"));
        EXPECT_TRUE(contains("--recreate_xfer=true"));
        EXPECT_TRUE(contains("--randomize_location_mode=blockaligned"));
    }

} // namespace
} // namespace nixlbench
