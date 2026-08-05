/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark/allocate_once.h"

#include "utils/utils.h"
#include "benchmark/allocate_once_worker.h"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <ostream>
#include <set>
#include <string_view>
#include <sys/stat.h>
#include <unordered_set>
#include <unistd.h>
#include <utility>

namespace nixlbench {
namespace {

    constexpr int invalid_arguments_exit_code = 2;
    constexpr size_t initialization_chunk = 1024 * 1024;
    constexpr mode_t managed_file_permissions = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;

    class fileHandle {
    public:
        explicit fileHandle(int fd) : fd_(fd) {}

        ~fileHandle() {
            if (fd_ >= 0) {
                ::close(fd_);
            }
        }

        fileHandle(const fileHandle &) = delete;
        fileHandle &
        operator=(const fileHandle &) = delete;

    private:
        int fd_;
    };

    bool
    initializeManagedFile(int fd, uint64_t size, size_t alignment, bool direct, std::ostream &err) {
        if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
            err << "Failed to resize managed file: " << strerror(errno) << '\n';
            return false;
        }

        const size_t chunk_size = std::min<size_t>(initialization_chunk, size);
        void *storage = nullptr;
        const int allocation_status = posix_memalign(&storage, alignment, chunk_size);
        if (allocation_status != 0 || storage == nullptr) {
            err << "Failed to allocate managed-file initialization buffer: "
                << strerror(allocation_status) << '\n';
            return false;
        }
        std::unique_ptr<void, decltype(&free)> chunk(storage, &free);
        memset(chunk.get(), XFERBENCH_TARGET_BUFFER_ELEMENT, chunk_size);

        uint64_t offset = 0;
        while (offset < size) {
            const size_t count = std::min<uint64_t>(chunk_size, size - offset);
            size_t written_total = 0;
            while (written_total < count) {
                const ssize_t written = pwrite(fd,
                                               static_cast<char *>(chunk.get()) + written_total,
                                               count - written_total,
                                               static_cast<off_t>(offset + written_total));
                if (written <= 0 ||
                    (direct && static_cast<size_t>(written) != count - written_total)) {
                    err << "Failed to initialize managed file at offset " << offset + written_total
                        << ": " << (written < 0 ? strerror(errno) : "short write") << '\n';
                    return false;
                }
                written_total += static_cast<size_t>(written);
            }
            offset += count;
        }
        return true;
    }

} // namespace

std::optional<std::vector<threadFileRegion>>
allocateOnceThreadRegions(const allocateOnceRequest &request, std::string &error) {
    if (request.files.empty() || request.common.threads < 1 || request.common.blockSize == 0) {
        error = "files, threads, and block size must be configured";
        return std::nullopt;
    }
    const uint64_t slots_per_file = request.fileSize / request.common.blockSize;
    std::vector<size_t> threads_per_file(request.files.size(), 0);
    for (int thread = 0; thread < request.common.threads; ++thread) {
        ++threads_per_file[static_cast<size_t>(thread) % request.files.size()];
    }

    std::vector<size_t> file_thread_index(request.files.size(), 0);
    std::vector<threadFileRegion> regions;
    regions.reserve(static_cast<size_t>(request.common.threads));
    for (int thread = 0; thread < request.common.threads; ++thread) {
        const size_t file_index = static_cast<size_t>(thread) % request.files.size();
        const size_t ordinal = file_thread_index[file_index]++;
        const size_t count = threads_per_file[file_index];
        const uint64_t first = slots_per_file * ordinal / count;
        const uint64_t end = slots_per_file * (ordinal + 1) / count;
        if (end - first < request.common.batchSize) {
            error = "each thread partition must contain at least --batch-size blocks";
            return std::nullopt;
        }
        regions.push_back({file_index, first, end - first});
    }
    return regions;
}

offsetSequence::offsetSequence(threadFileRegion region,
                               size_t batch_size,
                               const std::string &randomize_location_mode,
                               uint64_t seed)
    : region_(region),
      batchSize_(batch_size),
      randomize_(randomize_location_mode == XFERBENCH_RANDOMIZE_LOCATION_MODE_BLOCK_ALIGNED),
      nextSequentialSlot_(region.firstSlot),
      random_(seed) {}

std::vector<uint64_t>
offsetSequence::next() {
    std::vector<uint64_t> result;
    result.reserve(batchSize_);
    if (!randomize_) {
        for (size_t index = 0; index < batchSize_; ++index) {
            result.push_back(nextSequentialSlot_);
            const uint64_t relative = nextSequentialSlot_ - region_.firstSlot;
            nextSequentialSlot_ = region_.firstSlot + (relative + 1) % region_.slotCount;
        }
        return result;
    }

    std::unordered_set<uint64_t> selected;
    const uint64_t start = region_.slotCount - batchSize_;
    for (uint64_t candidate = start; candidate < region_.slotCount; ++candidate) {
        std::uniform_int_distribution<uint64_t> distribution(0, candidate);
        const uint64_t random_slot = distribution(random_);
        const uint64_t chosen =
            selected.find(random_slot) == selected.end() ? random_slot : candidate;
        selected.insert(chosen);
        result.push_back(region_.firstSlot + chosen);
    }
    return result;
}

bool
prepareAllocateOnceFiles(const allocateOnceRequest &request, std::ostream &err) {
    const long page_size_value = sysconf(_SC_PAGESIZE);
    if (page_size_value <= 0) {
        err << "Could not determine system page size\n";
        return false;
    }
    const size_t alignment = static_cast<size_t>(page_size_value);
    std::set<std::pair<dev_t, ino_t>> identities;

    for (const auto &path : request.files) {
        int flags =
            (request.managedFiles || request.common.operation == NIXL_WRITE ? O_RDWR : O_RDONLY) |
            O_LARGEFILE;
        if (request.managedFiles) {
            flags |= O_CREAT | O_NOFOLLOW;
        }
        if (request.direct) {
            flags |= O_DIRECT;
        }

        const int fd = open(path.c_str(), flags, managed_file_permissions);
        if (fd < 0) {
            err << "Failed to open " << path << ": " << strerror(errno) << '\n';
            return false;
        }
        fileHandle file(fd);

        struct stat info{};
        if (fstat(fd, &info) != 0) {
            err << "Failed to inspect " << path << ": " << strerror(errno) << '\n';
            return false;
        }
        if (request.managedFiles && !S_ISREG(info.st_mode)) {
            err << "Managed path must be a regular file: " << path << '\n';
            return false;
        }
        if (!identities.emplace(info.st_dev, info.st_ino).second) {
            err << "Backing files must refer to distinct files: " << path << '\n';
            return false;
        }
        if (request.managedFiles) {
            if (static_cast<uint64_t>(info.st_size) != request.fileSize &&
                !initializeManagedFile(fd, request.fileSize, alignment, request.direct, err)) {
                return false;
            }
        } else if (static_cast<uint64_t>(info.st_size) < request.fileSize) {
            err << "Explicit file " << path << " is smaller than --file-size\n";
            return false;
        }
    }
    return true;
}

namespace {

    std::string
    lower(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
        return value;
    }

    bool
    multiplyFits(size_t left, size_t right) {
        return left == 0 || right <= std::numeric_limits<size_t>::max() / left;
    }

    std::string
    memoryName(nixl_mem_t memory) {
        return memory == VRAM_SEG ? "VRAM" : "DRAM";
    }

    std::string
    operationName(nixl_xfer_op_t operation) {
        return operation == NIXL_READ ? "READ" : "WRITE";
    }

    bool
    validateRequest(allocateOnceRequest &request,
                    const fileOptions &file,
                    bool seed_provided,
                    std::ostream &err) {
        const auto fail = [&](const std::string &message) {
            err << "Error: " << message << '\n';
            return false;
        };

        if (request.fileSize == 0) {
            return fail("file size must be positive");
        }
        std::string file_error;
        if (!validateFileOptions(file, file_error)) {
            return fail(file_error);
        }
        if (file.numFiles > request.common.threads || request.common.threads % file.numFiles != 0) {
            return fail("--num-files must divide --threads and cannot exceed it");
        }
        if (request.common.checkConsistency && !file.filenames.empty()) {
            return fail("--check-consistency requires NIXLBench-managed files");
        }
        if (request.randomizeLocationMode == XFERBENCH_RANDOMIZE_LOCATION_MODE_NONE &&
            seed_provided) {
            return fail("--seed requires --randomize-location-mode=blockaligned");
        }
        if (request.fileSize % request.common.blockSize != 0) {
            return fail("--file-size must be an exact multiple of --block-size");
        }

        const size_t slots_per_file = request.fileSize / request.common.blockSize;
        const size_t threads_per_file = static_cast<size_t>(request.common.threads / file.numFiles);
        if (!multiplyFits(threads_per_file, request.common.batchSize) ||
            slots_per_file < threads_per_file * request.common.batchSize) {
            return fail(
                "each thread needs at least --batch-size block slots in its file partition");
        }
        if (!multiplyFits(request.fileSize, static_cast<size_t>(file.numFiles))) {
            return fail("total dataset size is too large for this platform");
        }
        if (!multiplyFits(request.common.blockSize, request.common.batchSize) ||
            !multiplyFits(request.common.blockSize * request.common.batchSize,
                          static_cast<size_t>(request.common.threads))) {
            return fail("working-memory size is too large for this platform");
        }

        if (file.direct) {
            const long page_size = sysconf(_SC_PAGESIZE);
            if (page_size <= 0) {
                return fail("could not determine the system page size for direct I/O");
            }
            const size_t alignment = static_cast<size_t>(page_size);
            if (request.fileSize % alignment != 0 || request.common.blockSize % alignment != 0) {
                return fail(
                    "--direct requires file and block sizes aligned to the system page size");
            }
        }

        request.files = allocateOnceFileNames(file);
        request.managedFiles = file.filenames.empty();
        request.direct = file.direct;
        return true;
    }

} // namespace

struct allocateOnceScenario::implementation {
    scenarioOptions common;
    std::string fileSize;
    std::string randomizeLocationMode = XFERBENCH_RANDOMIZE_LOCATION_MODE_BLOCK_ALIGNED;
    uint64_t seed = 0;
    fileOptions file;
    CLI::App *command = nullptr;
    CLI::Option *seedOption = nullptr;
    scenario_plugin_bindings_t plugins;
    allocateOnceRequest request;
};

allocateOnceScenario::allocateOnceScenario()
    : implementation_(std::make_unique<implementation>()) {}

allocateOnceScenario::~allocateOnceScenario() = default;

bool
supportsAllocateOnce(const pluginMetadata &metadata) {
    return hasMemoryType(metadata, FILE_SEG) &&
        (hasMemoryType(metadata, DRAM_SEG) || hasMemoryType(metadata, VRAM_SEG));
}

std::vector<std::filesystem::path>
allocateOnceFileNames(const fileOptions &file) {
    std::vector<std::filesystem::path> names;
    if (!file.filenames.empty()) {
        for (const auto &name : splitFileNames(file.filenames)) {
            names.emplace_back(name);
        }
        return names;
    }

    const std::filesystem::path directory =
        file.path.empty() ? std::filesystem::current_path() : std::filesystem::path(file.path);
    names.reserve(static_cast<size_t>(file.numFiles));
    for (int index = 0; index < file.numFiles; ++index) {
        names.push_back(directory / ("nixlbench_allocate_once_" + std::to_string(index) + ".dat"));
    }
    return names;
}

size_t
allocateOnceWorkingMemory(const allocateOnceRequest &request) {
    return request.common.blockSize * request.common.batchSize *
        static_cast<size_t>(request.common.threads);
}

int
allocateOnceScenario::addCommand(CLI::App &scenario,
                                 const std::vector<pluginMetadata> &metadata,
                                 std::ostream &err) {
    implementation_->command = scenario.add_subcommand(
        "allocate-once", "Reuse fixed registered files while transferring at changing offsets");
    implementation_->command->require_subcommand(1);

    implementation_->command
        ->add_option("--file-size", implementation_->fileSize, "Logical size of each backing file")
        ->required()
        ->group("Allocate-once options");
    implementation_->command
        ->add_option("--randomize-location-mode",
                     implementation_->randomizeLocationMode,
                     "Offset randomization: none or blockaligned")
        ->check(CLI::IsMember({XFERBENCH_RANDOMIZE_LOCATION_MODE_NONE,
                               XFERBENCH_RANDOMIZE_LOCATION_MODE_BLOCK_ALIGNED},
                              CLI::ignore_case))
        ->group("Allocate-once options");
    implementation_->seedOption =
        implementation_->command
            ->add_option("--seed",
                         implementation_->seed,
                         "Seed for reproducible block-aligned randomization")
            ->group("Allocate-once options");
    addCommonScenarioOptions(*implementation_->command, implementation_->common);

    const int plugin_status = addScenarioPluginCommands(
        *implementation_->command, metadata, supportsAllocateOnce, implementation_->plugins, err);
    if (plugin_status != EXIT_SUCCESS) {
        return plugin_status;
    }
    for (const auto &binding : implementation_->plugins) {
        addFileScenarioOptions(*binding->command, implementation_->file);
    }
    return EXIT_SUCCESS;
}

bool
allocateOnceScenario::selected() const {
    return implementation_->command != nullptr && implementation_->command->parsed();
}

int
allocateOnceScenario::finalize(std::ostream &err) {
    const auto *selected_plugin = selectedScenarioPlugin(implementation_->plugins);
    if (selected_plugin == nullptr) {
        err << "Error: allocate-once requires an installed compatible FILE_SEG plugin\n";
        return invalid_arguments_exit_code;
    }

    auto &request = implementation_->request;
    if (!resolveCommonScenarioOptions(implementation_->common,
                                      selected_plugin->metadata,
                                      selected_plugin->overrides,
                                      request.common,
                                      err)) {
        return invalid_arguments_exit_code;
    }

    std::string size_error;
    const auto parsed_file_size = parseHumanSize(implementation_->fileSize, size_error);
    if (!parsed_file_size) {
        err << "Error: invalid file size '" << implementation_->fileSize << "': " << size_error
            << '\n';
        return invalid_arguments_exit_code;
    }
    request.fileSize = *parsed_file_size;
    request.randomizeLocationMode = lower(implementation_->randomizeLocationMode);
    request.seed = implementation_->seed;

    if (!validateRequest(
            request, implementation_->file, implementation_->seedOption->count() != 0, err)) {
        return invalid_arguments_exit_code;
    }
    return EXIT_SUCCESS;
}

void
allocateOnceScenario::printPlan(std::ostream &out) const {
    const auto &request = implementation_->request;
    const size_t total_dataset = request.fileSize * request.files.size();
    out << "Resolved NIXLBench scenario\n"
        << "  scenario: allocate-once\n"
        << "  backend: " << request.common.pluginName
        << "\n  initiator memory: " << memoryName(request.common.initiatorMemory)
        << "\n  dataset policy: "
        << (request.managedFiles ? "managed, initialize if needed, keep after run" :
                                   "explicit files, reuse only")
        << "\n  files: " << request.files.size() << "\n  resolved file paths:";
    for (const auto &file : request.files) {
        out << "\n    " << file;
    }
    out << "\n  size per file: " << formatSize(request.fileSize)
        << "\n  total dataset: " << formatSize(total_dataset)
        << "\n  block size: " << formatSize(request.common.blockSize)
        << "\n  batch size: " << request.common.batchSize << " blocks"
        << "\n  worker threads: " << request.common.threads
        << "\n  working memory: " << formatSize(allocateOnceWorkingMemory(request))
        << "\n  operation: " << operationName(request.common.operation)
        << "\n  randomize location mode: " << request.randomizeLocationMode;
    if (request.randomizeLocationMode == XFERBENCH_RANDOMIZE_LOCATION_MODE_BLOCK_ALIGNED) {
        out << ", seed " << request.seed;
    }
    out << "\n  warmup requests: " << request.common.warmupIterations << " per thread"
        << "\n  timed requests: " << request.common.iterations << " per thread, "
        << static_cast<uint64_t>(request.common.iterations) *
            static_cast<uint64_t>(request.common.threads)
        << " aggregate"
        << "\n  direct I/O: " << (request.direct ? "enabled" : "disabled")
        << "\n  consistency check: " << (request.common.checkConsistency ? "enabled" : "disabled")
        << "\n  lifecycle: open, allocate, and register once; release each transfer request"
        << "\n  execution: shared NIXLBench worker facilities"
        << "\n  timing: request preparation, post, transfer latency, and throughput\n"
        << "  plugin parameters:\n";
    for (const auto &key : sortedParameterKeys(request.common.pluginParameters)) {
        out << "    " << key << ": " << request.common.pluginParameters.at(key) << '\n';
    }
    if (request.common.dryRun) {
        out << "Dry run: no backing file was opened and no transfer buffer, registration, or "
               "transfer request was created.\n";
    }
}

bool
allocateOnceScenario::dryRun() const {
    return implementation_->request.common.dryRun;
}

bool
allocateOnceScenario::prepare(std::ostream &err) const {
    return prepareAllocateOnceFiles(implementation_->request, err);
}

legacyWorkerConfig
allocateOnceScenario::legacyWorkerConfiguration() const {
    const auto &request = implementation_->request;
    legacyWorkerConfig config;
    config.common = request.common;
    config.workingMemory = allocateOnceWorkingMemory(request);
    config.targetMemory = FILE_SEG;
    config.recreateTransferRequest = true;
    config.fileNames.reserve(request.files.size());
    for (const auto &file : request.files) {
        config.fileNames.push_back(file.string());
    }
    config.storageDirect = request.direct;
    config.randomizeLocationMode = request.randomizeLocationMode;
    config.randomizeLocationSeed = request.seed;
    return config;
}

std::unique_ptr<xferBenchWorker>
allocateOnceScenario::createWorker(const std::vector<std::string> &devices) const {
    return std::make_unique<xferBenchNixlAllocateOnceWorker>(
        devices, implementation_->request.common.pluginParameters, implementation_->request);
}

} // namespace nixlbench
