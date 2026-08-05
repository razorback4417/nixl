/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/scenario_cli.h"

#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <vector>

namespace nixlbench {
namespace {

    class Arguments {
    public:
        Arguments(std::initializer_list<const char *> values) {
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

    PluginMetadata
    posixMetadata() {
        return {"POSIX",
                {DRAM_SEG, FILE_SEG},
                {{"future_parameter", "default"}, {"ios_pool_size", "4096"}}};
    }

    PluginMetadata
    futureFileMetadata() {
        return {
            "FUTURE-FILE", {VRAM_SEG, DRAM_SEG, FILE_SEG}, {{"provider-key", "provider default"}}};
    }

    int
    parse(Arguments &arguments,
          const std::vector<PluginMetadata> &metadata,
          AllocateOnceRequest &request,
          std::ostringstream &out,
          std::ostringstream &err,
          bool &help) {
        return parseAllocateOnceCommand(
            arguments.argc(), arguments.argv(), metadata, request, help, out, err);
    }

    TEST(ScenarioCliDispatchTest, OnlyExplicitScenarioSelectsTheScenarioParser) {
        Arguments scenario{"nixlbench", "scenario", "allocate-once"};
        EXPECT_TRUE(isScenarioCommand(scenario.argc(), scenario.argv()));

        Arguments raw{"nixlbench", "raw", "posix"};
        EXPECT_FALSE(isScenarioCommand(raw.argc(), raw.argv()));

        Arguments legacy{"nixlbench", "--backend=POSIX"};
        EXPECT_FALSE(isScenarioCommand(legacy.argc(), legacy.argv()));
    }

    TEST(ScenarioMetadataTest, CompatibilityComesFromAdvertisedMemoryTypes) {
        EXPECT_TRUE(supportsAllocateOnce(posixMetadata()));
        EXPECT_TRUE(supportsAllocateOnce(futureFileMetadata()));
        EXPECT_FALSE(supportsAllocateOnce({"UCX", {DRAM_SEG}, {}}));
        EXPECT_FALSE(supportsAllocateOnce({"FILE_ONLY", {FILE_SEG}, {}}));
    }

    TEST(ScenarioParserTest, AcceptsCompatibleFuturePluginAndPreservesOpaqueParameters) {
        Arguments arguments{"nixlbench",
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
        AllocateOnceRequest request;
        bool help = false;
        std::ostringstream out;
        std::ostringstream err;

        ASSERT_EQ(parse(arguments, {futureFileMetadata()}, request, out, err, help), 0)
            << err.str();
        EXPECT_EQ(request.plugin_name, "FUTURE-FILE");
        EXPECT_EQ(request.plugin_parameters.at("provider-key"), "Exact Value");
        EXPECT_EQ(request.initiator_memory, VRAM_SEG);
    }

    TEST(ScenarioParserTest, ScenarioOptionsWorkBeforeOrAfterPluginSelection) {
        Arguments before{"nixlbench",
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
        Arguments after{"nixlbench",
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
            AllocateOnceRequest request;
            bool help = false;
            std::ostringstream out;
            std::ostringstream err;
            ASSERT_EQ(parse(*arguments, {posixMetadata()}, request, out, err, help), 0)
                << err.str();
            EXPECT_EQ(request.threads, 2);
            EXPECT_EQ(request.files.size(), 2U);
        }
    }

    TEST(ScenarioParserTest, RejectsInvalidCapacityOwnershipAndMemoryRequests) {
        const auto expect_failure = [](Arguments &arguments, const std::string &message) {
            AllocateOnceRequest request;
            bool help = false;
            std::ostringstream out;
            std::ostringstream err;
            EXPECT_NE(parse(arguments, {posixMetadata()}, request, out, err, help), 0);
            EXPECT_NE(err.str().find(message), std::string::npos) << err.str();
        };

        Arguments too_small{"nixlbench",
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

        Arguments fractional_block{"nixlbench",
                                   "scenario",
                                   "allocate-once",
                                   "--file-size",
                                   "10KiB",
                                   "--block-size",
                                   "4KiB",
                                   "posix"};
        expect_failure(fractional_block, "exact multiple");

        Arguments mixed_ownership{"nixlbench",
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

        Arguments unsupported_memory{"nixlbench",
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
    }

    TEST(ScenarioParserTest, ExplicitMemoryDoesNotSilentlyFallBack) {
        Arguments arguments{"nixlbench",
                            "scenario",
                            "allocate-once",
                            "--file-size",
                            "1MiB",
                            "--block-size",
                            "4KiB",
                            "--initiator-memory",
                            "dram",
                            "future-file"};
        AllocateOnceRequest request;
        bool help = false;
        std::ostringstream out;
        std::ostringstream err;

        ASSERT_EQ(parse(arguments, {futureFileMetadata()}, request, out, err, help), 0)
            << err.str();
        EXPECT_EQ(request.initiator_memory, DRAM_SEG);
    }

    TEST(ScenarioPlanTest, SeparatesDatasetSizeFromBoundedTransferMemory) {
        AllocateOnceRequest request;
        request.plugin_name = "POSIX";
        request.plugin_parameters = {{"ios_pool_size", "4096"}};
        request.file_size = 1024 * 1024;
        request.block_size = 4096;
        request.batch_size = 4;
        request.threads = 3;
        request.files = {"/tmp/nixlbench_allocate_once_0.dat"};
        request.initiator_memory = DRAM_SEG;
        request.dry_run = true;

        EXPECT_EQ(allocateOnceWorkingMemory(request), 3U * 4U * 4096U);

        std::ostringstream out;
        printAllocateOncePlan(request, out);
        EXPECT_NE(out.str().find("size per file: 1 MiB"), std::string::npos);
        EXPECT_NE(out.str().find("working memory: 48 KiB"), std::string::npos);
        EXPECT_NE(out.str().find("open, allocate, and register once"), std::string::npos);
        EXPECT_NE(out.str().find("no backing file was opened"), std::string::npos);
    }

    TEST(ScenarioPlanTest, MapsWorkloadIntentIntoTheSharedWorkerConfiguration) {
        AllocateOnceRequest request;
        request.plugin_name = "FUTURE-FILE";
        request.file_size = 1024 * 1024;
        request.block_size = 4096;
        request.batch_size = 4;
        request.threads = 2;
        request.iterations = 7;
        request.warmup_iterations = 3;
        request.files = {"/tmp/file-0", "/tmp/file-1"};
        request.initiator_memory = VRAM_SEG;

        const auto arguments = allocateOnceBenchmarkArguments(request, "nixlbench");
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
    }

} // namespace
} // namespace nixlbench
