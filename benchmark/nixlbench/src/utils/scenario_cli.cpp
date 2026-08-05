/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/scenario_cli.h"

#include "utils/utils.h"

#include <CLI/CLI.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string_view>
#include <unistd.h>
#include <utility>

namespace nixlbench {
namespace {

    constexpr int kInvalidArgumentsExitCode = 2;

    struct PluginBinding {
        const PluginMetadata *metadata = nullptr;
        CLI::App *command = nullptr;
        std::vector<std::pair<std::string, std::string>> overrides;
    };

    enum class RequestedMemory {
        Auto,
        Dram,
        Vram,
    };

    std::string
    upper(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::toupper(ch));
        });
        return value;
    }

    std::string
    lower(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
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
    validateRequest(AllocateOnceRequest &request,
                    const FileOptions &file,
                    const PluginMetadata &metadata,
                    RequestedMemory requested_memory,
                    bool seed_provided,
                    std::ostream &err) {
        const auto fail = [&](const std::string &message) {
            err << "Error: " << message << '\n';
            return false;
        };

        if (request.file_size == 0 || request.block_size == 0) {
            return fail("file size and block size must be positive");
        }
        if (request.batch_size == 0 || request.threads < 1 || request.iterations < 1 ||
            request.warmup_iterations < 0) {
            return fail("batch size, threads, and iterations must be positive; warmup iterations "
                        "may be zero");
        }
        if (request.iterations > std::numeric_limits<int>::max() / request.threads ||
            request.warmup_iterations > std::numeric_limits<int>::max() / request.threads) {
            return fail("aggregate iteration count is too large");
        }
        std::string file_error;
        if (!validateFileOptions(file, file_error)) {
            return fail(file_error);
        }
        if (file.num_files > request.threads || request.threads % file.num_files != 0) {
            return fail("--num-files must divide --threads and cannot exceed it");
        }
        if (request.check_consistency && !file.filenames.empty()) {
            return fail("--check-consistency requires NIXLBench-managed files");
        }
        if (request.offset_mode == OffsetMode::Sequential && seed_provided) {
            return fail("--seed is only valid with --offset-mode random");
        }
        if (request.file_size % request.block_size != 0) {
            return fail("--file-size must be an exact multiple of --block-size");
        }

        const size_t slots_per_file = request.file_size / request.block_size;
        const size_t threads_per_file = static_cast<size_t>(request.threads / file.num_files);
        if (!multiplyFits(threads_per_file, request.batch_size) ||
            slots_per_file < threads_per_file * request.batch_size) {
            return fail(
                "each thread needs at least --batch-size block slots in its file partition");
        }
        if (!multiplyFits(request.file_size, static_cast<size_t>(file.num_files))) {
            return fail("total dataset size is too large for this platform");
        }
        if (!multiplyFits(request.block_size, request.batch_size) ||
            !multiplyFits(request.block_size * request.batch_size,
                          static_cast<size_t>(request.threads))) {
            return fail("working-memory size is too large for this platform");
        }

        if (file.direct) {
            const long page_size = sysconf(_SC_PAGESIZE);
            if (page_size <= 0) {
                return fail("could not determine the system page size for direct I/O");
            }
            const size_t alignment = static_cast<size_t>(page_size);
            if (request.file_size % alignment != 0 || request.block_size % alignment != 0) {
                return fail(
                    "--direct requires file and block sizes aligned to the system page size");
            }
        }

        const bool supports_dram = hasMemoryType(metadata, DRAM_SEG);
        const bool supports_vram = hasMemoryType(metadata, VRAM_SEG);
        if (requested_memory == RequestedMemory::Dram && !supports_dram) {
            return fail(metadata.name + " does not advertise DRAM_SEG");
        }
        if (requested_memory == RequestedMemory::Vram && !supports_vram) {
            return fail(metadata.name + " does not advertise VRAM_SEG");
        }
        if (requested_memory == RequestedMemory::Vram ||
            (requested_memory == RequestedMemory::Auto && supports_vram)) {
            request.initiator_memory = VRAM_SEG;
        } else if (supports_dram) {
            request.initiator_memory = DRAM_SEG;
        } else {
            return fail(metadata.name + " has no supported initiator memory type");
        }
        request.files = allocateOnceFileNames(file);
        request.managed_files = file.filenames.empty();
        request.direct = file.direct;
        return true;
    }

} // namespace

bool
isScenarioCommand(int argc, char *argv[]) {
    return argc > 1 && std::string_view(argv[1]) == "scenario";
}

bool
supportsAllocateOnce(const PluginMetadata &metadata) {
    return hasMemoryType(metadata, FILE_SEG) &&
        (hasMemoryType(metadata, DRAM_SEG) || hasMemoryType(metadata, VRAM_SEG));
}

std::vector<std::filesystem::path>
allocateOnceFileNames(const FileOptions &file) {
    std::vector<std::filesystem::path> names;
    if (!file.filenames.empty()) {
        for (const auto &name : splitFileNames(file.filenames)) {
            names.emplace_back(name);
        }
        return names;
    }

    const std::filesystem::path directory =
        file.path.empty() ? std::filesystem::current_path() : std::filesystem::path(file.path);
    names.reserve(static_cast<size_t>(file.num_files));
    for (int index = 0; index < file.num_files; ++index) {
        names.push_back(directory / ("nixlbench_allocate_once_" + std::to_string(index) + ".dat"));
    }
    return names;
}

int
parseAllocateOnceCommand(int argc,
                         char *argv[],
                         const std::vector<PluginMetadata> &metadata,
                         AllocateOnceRequest &request,
                         bool &help_requested,
                         std::ostream &out,
                         std::ostream &err) {
    help_requested = false;
    std::string file_size;
    std::string block_size;
    std::string operation = "write";
    std::string offset_mode = "random";
    std::string initiator_memory = "auto";
    FileOptions file;

    CLI::App app("NIXL data-transfer benchmark");
    app.require_subcommand(1);
    auto *scenario = app.add_subcommand("scenario", "Run a modeled transfer workload");
    scenario->require_subcommand(1);
    auto *allocate_once = scenario->add_subcommand(
        "allocate-once", "Reuse fixed registered files while transferring at changing offsets");
    allocate_once->require_subcommand(1);

    allocate_once->add_option("--file-size", file_size, "Logical size of each backing file")
        ->required()
        ->group("Allocate-once options");
    allocate_once->add_option("--block-size", block_size, "Bytes in each transferred KV block")
        ->required()
        ->group("Allocate-once options");
    allocate_once
        ->add_option("--batch-size", request.batch_size, "Block descriptors in each request")
        ->group("Allocate-once options");
    allocate_once->add_option("--threads", request.threads, "Parallel transfer threads")
        ->group("Allocate-once options");
    allocate_once->add_option("--iterations", request.iterations, "Timed requests per thread")
        ->group("Allocate-once options");
    allocate_once
        ->add_option(
            "--warmup-iterations", request.warmup_iterations, "Untimed warmup requests per thread")
        ->group("Allocate-once options");
    allocate_once->add_option("--operation", operation, "Transfer direction: read or write")
        ->check(CLI::IsMember({"read", "write"}, CLI::ignore_case))
        ->group("Allocate-once options");
    allocate_once->add_option("--offset-mode", offset_mode, "Offset pattern: random or sequential")
        ->check(CLI::IsMember({"random", "sequential"}, CLI::ignore_case))
        ->group("Allocate-once options");
    auto *seed_option =
        allocate_once->add_option("--seed", request.seed, "Seed for reproducible random offsets")
            ->group("Allocate-once options");
    allocate_once
        ->add_option(
            "--initiator-memory", initiator_memory, "Local buffer placement: auto, dram, or vram")
        ->check(CLI::IsMember({"auto", "dram", "vram"}, CLI::ignore_case))
        ->group("Allocate-once options");
    allocate_once
        ->add_flag("--check-consistency",
                   request.check_consistency,
                   "Validate transferred bytes outside transfer timing")
        ->group("Allocate-once options");
    allocate_once
        ->add_flag("--dry-run", request.dry_run, "Print the resolved plan without executing")
        ->group("Allocate-once options");

    std::vector<std::unique_ptr<PluginBinding>> bindings;
    std::set<std::string> command_names;
    bindings.reserve(metadata.size());
    for (const auto &entry : metadata) {
        if (!supportsAllocateOnce(entry)) {
            continue;
        }
        const std::string command_name = lower(entry.name);
        if (!command_names.insert(command_name).second) {
            err << "Error: installed plugin names are ambiguous when used as CLI subcommands: "
                << command_name << '\n';
            return kInvalidArgumentsExitCode;
        }
        auto binding = std::make_unique<PluginBinding>();
        binding->metadata = &entry;
        binding->command = allocate_once->add_subcommand(
            command_name, "Run the installed " + entry.name + " FILE_SEG backend");
        binding->command->fallthrough();
        binding->command->footer(
            "Allocate-once options may be used before or after this plugin subcommand.");
        binding->command
            ->add_option("--path", file.path, "Directory for deterministic NIXLBench-managed files")
            ->group("FILE_SEG resource options");
        binding->command
            ->add_option("--filenames", file.filenames, "Comma-separated existing file names")
            ->group("FILE_SEG resource options");
        binding->command->add_option("--num-files", file.num_files, "Number of backing files")
            ->group("FILE_SEG resource options");
        binding->command->add_flag("--direct", file.direct, "Use direct file opening")
            ->group("FILE_SEG resource options");
        if (!entry.parameters.empty()) {
            binding->command
                ->add_option("--plugin-param",
                             binding->overrides,
                             pluginParameterDescription(entry.parameters))
                ->check(CLI::IsMember(sortedParameterKeys(entry.parameters))
                            .description("")
                            .application_index(0))
                ->type_name("KEY VALUE")
                ->group("Plugin initialization parameters");
        }
        bindings.push_back(std::move(binding));
    }

    try {
        app.parse(argc, argv);
    }
    catch (const CLI::CallForHelp &exception) {
        help_requested = true;
        return app.exit(exception, out, err);
    }
    catch (const CLI::ParseError &exception) {
        return app.exit(exception, out, err);
    }

    PluginBinding *selected = nullptr;
    for (auto &binding : bindings) {
        if (binding->command->parsed()) {
            selected = binding.get();
            break;
        }
    }
    if (selected == nullptr) {
        err << "Error: allocate-once requires an installed compatible FILE_SEG plugin\n";
        return kInvalidArgumentsExitCode;
    }

    request.plugin_name = selected->metadata->name;
    request.plugin_parameters = selected->metadata->parameters;
    for (const auto &[key, value] : selected->overrides) {
        request.plugin_parameters[key] = value;
    }

    std::string size_error;
    const auto parsed_file_size = parseHumanSize(file_size, size_error);
    if (!parsed_file_size) {
        err << "Error: invalid file size '" << file_size << "': " << size_error << '\n';
        return kInvalidArgumentsExitCode;
    }
    const auto parsed_block_size = parseHumanSize(block_size, size_error);
    if (!parsed_block_size) {
        err << "Error: invalid block size '" << block_size << "': " << size_error << '\n';
        return kInvalidArgumentsExitCode;
    }
    request.file_size = *parsed_file_size;
    request.block_size = *parsed_block_size;
    request.operation = upper(operation) == "READ" ? NIXL_READ : NIXL_WRITE;
    request.offset_mode =
        upper(offset_mode) == "RANDOM" ? OffsetMode::Random : OffsetMode::Sequential;
    const std::string memory = upper(initiator_memory);
    const RequestedMemory requested_memory = memory == "DRAM" ? RequestedMemory::Dram :
        memory == "VRAM"                                      ? RequestedMemory::Vram :
                                                                RequestedMemory::Auto;

    if (!validateRequest(
            request, file, *selected->metadata, requested_memory, seed_option->count() != 0, err)) {
        return kInvalidArgumentsExitCode;
    }
    return EXIT_SUCCESS;
}

size_t
allocateOnceWorkingMemory(const AllocateOnceRequest &request) {
    return request.block_size * request.batch_size * static_cast<size_t>(request.threads);
}

std::vector<std::string>
allocateOnceBenchmarkArguments(const AllocateOnceRequest &request,
                               const std::string &program_name) {
    const auto boolean = [](bool value) { return value ? "true" : "false"; };
    std::ostringstream filenames;
    for (size_t index = 0; index < request.files.size(); ++index) {
        if (index != 0) {
            filenames << ',';
        }
        filenames << request.files[index].string();
    }

    return {
        program_name,
        std::string("--worker_type=") + XFERBENCH_WORKER_NIXL,
        "--backend=" + request.plugin_name,
        std::string("--initiator_seg_type=") +
            (request.initiator_memory == VRAM_SEG ? XFERBENCH_SEG_TYPE_VRAM :
                                                    XFERBENCH_SEG_TYPE_DRAM),
        std::string("--target_seg_type=") + XFERBENCH_SEG_TYPE_FILE,
        std::string("--op_type=") +
            (request.operation == NIXL_READ ? XFERBENCH_OP_READ : XFERBENCH_OP_WRITE),
        "--check_consistency=" + std::string(boolean(request.check_consistency)),
        "--total_buffer_size=" + std::to_string(allocateOnceWorkingMemory(request)),
        "--start_block_size=" + std::to_string(request.block_size),
        "--max_block_size=" + std::to_string(request.block_size),
        "--start_batch_size=" + std::to_string(request.batch_size),
        "--max_batch_size=" + std::to_string(request.batch_size),
        "--num_iter=" + std::to_string(request.iterations * request.threads),
        "--warmup_iter=" + std::to_string(request.warmup_iterations * request.threads),
        "--num_threads=" + std::to_string(request.threads),
        "--large_blk_iter_ftr=1",
        "--pipeline_depth=1",
        "--recreate_xfer=true",
        "--filenames=" + filenames.str(),
        "--num_files=" + std::to_string(request.files.size()),
        "--storage_enable_direct=" + std::string(boolean(request.direct)),
    };
}

void
printAllocateOncePlan(const AllocateOnceRequest &request, std::ostream &out) {
    const size_t total_dataset = request.file_size * request.files.size();
    out << "Resolved NIXLBench scenario\n"
        << "  scenario: allocate-once\n"
        << "  backend: " << request.plugin_name
        << "\n  initiator memory: " << memoryName(request.initiator_memory)
        << "\n  dataset policy: "
        << (request.managed_files ? "managed, initialize if needed, keep after run" :
                                    "explicit files, reuse only")
        << "\n  files: " << request.files.size() << "\n  resolved file paths:";
    for (const auto &file : request.files) {
        out << "\n    " << file;
    }
    out << "\n  size per file: " << formatSize(request.file_size)
        << "\n  total dataset: " << formatSize(total_dataset)
        << "\n  block size: " << formatSize(request.block_size)
        << "\n  batch size: " << request.batch_size << " blocks"
        << "\n  worker threads: " << request.threads
        << "\n  working memory: " << formatSize(allocateOnceWorkingMemory(request))
        << "\n  operation: " << operationName(request.operation) << "\n  offsets: "
        << (request.offset_mode == OffsetMode::Random ?
                "random, seed " + std::to_string(request.seed) :
                "sequential")
        << "\n  warmup requests: " << request.warmup_iterations << " per thread"
        << "\n  timed requests: " << request.iterations << " per thread, "
        << static_cast<uint64_t>(request.iterations) * static_cast<uint64_t>(request.threads)
        << " aggregate"
        << "\n  direct I/O: " << (request.direct ? "enabled" : "disabled")
        << "\n  consistency check: " << (request.check_consistency ? "enabled" : "disabled")
        << "\n  lifecycle: open, allocate, and register once; release each transfer request"
        << "\n  execution: shared NIXLBench worker"
        << "\n  timing: request preparation, post, transfer latency, and throughput\n"
        << "  plugin parameters:\n";
    for (const auto &key : sortedParameterKeys(request.plugin_parameters)) {
        out << "    " << key << ": " << request.plugin_parameters.at(key) << '\n';
    }
    if (request.dry_run) {
        out << "Dry run: no backing file was opened and no transfer buffer, registration, or "
               "transfer request was created.\n";
    }
}

ScenarioCommandResult
prepareScenarioCommand(int argc, char *argv[], std::ostream &out, std::ostream &err) {
    std::string discovery_error;
    auto metadata = discoverPluginMetadata(discovery_error);
    if (!metadata) {
        err << "Error: " << discovery_error << '\n';
        return {EXIT_FAILURE, false};
    }

    AllocateOnceRequest request;
    bool help_requested = false;
    const int parse_status =
        parseAllocateOnceCommand(argc, argv, *metadata, request, help_requested, out, err);
    if (parse_status != EXIT_SUCCESS || help_requested) {
        return {parse_status, false};
    }

    printAllocateOncePlan(request, out);
    return {EXIT_SUCCESS, !request.dry_run, std::move(request)};
}

} // namespace nixlbench
