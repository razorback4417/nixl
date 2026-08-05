/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/scenario_cli.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <set>

namespace nixlbench {
namespace {

    TEST(AllocateOnceRegionsTest, AssignsThreadsRoundRobinToDisjointFilePartitions) {
        AllocateOnceRequest request;
        request.file_size = 10 * 4096;
        request.block_size = 4096;
        request.batch_size = 2;
        request.threads = 4;
        request.files = {"/tmp/file-0", "/tmp/file-1"};

        std::string error;
        const auto regions = allocateOnceThreadRegions(request, error);
        ASSERT_TRUE(regions) << error;
        ASSERT_EQ(regions->size(), 4U);
        EXPECT_EQ((*regions)[0].file_index, 0U);
        EXPECT_EQ((*regions)[1].file_index, 1U);
        EXPECT_EQ((*regions)[2].file_index, 0U);
        EXPECT_EQ((*regions)[3].file_index, 1U);
        EXPECT_EQ((*regions)[0].first_slot, 0U);
        EXPECT_EQ((*regions)[0].slot_count, 5U);
        EXPECT_EQ((*regions)[2].first_slot, 5U);
        EXPECT_EQ((*regions)[2].slot_count, 5U);
    }

    TEST(OffsetSequenceTest, RandomBatchesAreSeededUniqueAndRemainInTheThreadRegion) {
        const ThreadFileRegion region{0, 10, 8};
        OffsetSequence first(region, 4, OffsetMode::Random, 1234);
        OffsetSequence second(region, 4, OffsetMode::Random, 1234);

        const auto first_batch = first.next();
        const auto second_batch = second.next();
        EXPECT_EQ(first_batch, second_batch);
        EXPECT_EQ(std::set<uint64_t>(first_batch.begin(), first_batch.end()).size(),
                  first_batch.size());
        EXPECT_TRUE(std::all_of(first_batch.begin(), first_batch.end(), [](uint64_t slot) {
            return slot >= 10 && slot < 18;
        }));
    }

    TEST(OffsetSequenceTest, SequentialBatchesWrapInsideTheThreadRegion) {
        OffsetSequence offsets({0, 7, 5}, 3, OffsetMode::Sequential, 0);
        EXPECT_EQ(offsets.next(), (std::vector<uint64_t>{7, 8, 9}));
        EXPECT_EQ(offsets.next(), (std::vector<uint64_t>{10, 11, 7}));
    }

    TEST(AllocateOnceFileNamesTest, ManagedNamesAreScenarioOwnedAndDeterministic) {
        FileOptions file;
        file.path = "/tmp/scenario";
        file.num_files = 2;

        const auto names = allocateOnceFileNames(file);
        ASSERT_EQ(names.size(), 2U);
        EXPECT_EQ(names[0], "/tmp/scenario/nixlbench_allocate_once_0.dat");
        EXPECT_EQ(names[1], "/tmp/scenario/nixlbench_allocate_once_1.dat");
    }

    TEST(AllocateOnceFileNamesTest, PreservesExplicitFileNames) {
        FileOptions file;
        file.filenames = "/tmp/name one,/tmp/name-two";
        file.num_files = 2;

        const auto names = allocateOnceFileNames(file);
        ASSERT_EQ(names.size(), 2U);
        EXPECT_EQ(names[0], "/tmp/name one");
        EXPECT_EQ(names[1], "/tmp/name-two");
    }

} // namespace
} // namespace nixlbench
