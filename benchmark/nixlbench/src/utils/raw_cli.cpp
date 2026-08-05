/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/raw_cli.h"

#include "utils/utils.h"

#include <CLI/CLI.hpp>
#include <nixl.h>

#include <algorithm>
#include <cctype>
#include <string_view>
#include <utility>

namespace nixlbench {
namespace {

    constexpr int invalid_arguments_exit_code = 2;

    std::string
    upper(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::toupper(ch));
        });
        return value;
    }

    bool
    validateRawOptions(const rawOptions &raw, std::ostream &err) {
        const auto fail = [&](const std::string &message) {
            err << "Error: " << message << '\n';
            return false;
        };

        if (raw.operation != XFERBENCH_OP_READ && raw.operation != XFERBENCH_OP_WRITE) {
            return fail("--operation must be read or write");
        }
        if (raw.threads < 1 || raw.iterations < 1 || raw.warmupIterations < 0 ||
            raw.pipelineDepth < 1) {
            return fail(
                "threads, iterations, and pipeline depth must be positive; warmup may be zero");
        }
        if (raw.startBlockSize == 0 || raw.maxBlockSize < raw.startBlockSize) {
            return fail("block sizes must be positive and max must be at least start");
        }
        if (raw.startBatchSize == 0 || raw.maxBatchSize < raw.startBatchSize) {
            return fail("batch sizes must be positive and max must be at least start");
        }
        return true;
    }

} // namespace

void
printRawPosixPlan(const rawPosixRequest &request,
                  const pluginMetadata &metadata,
                  int normalized_iterations,
                  int normalized_warmup_iterations,
                  std::ostream &out) {
    out << "Resolved NIXLBench plan\n"
        << "  command: raw posix\n"
        << "  backend: " << metadata.name << "\n"
        << "  memory types: ";
    auto memory_types = metadata.memoryTypes;
    std::sort(memory_types.begin(), memory_types.end());
    for (size_t i = 0; i < memory_types.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << nixlEnumStrings::memTypeStr(memory_types[i]);
    }
    out << "\n  benchmark options:\n"
        << "    operation: " << request.raw.operation
        << "\n    total buffer: " << formatSize(request.raw.totalBufferSize)
        << "\n    block sizes: " << formatSize(request.raw.startBlockSize) << " .. "
        << formatSize(request.raw.maxBlockSize)
        << "\n    batch sizes: " << request.raw.startBatchSize << " .. "
        << request.raw.maxBatchSize;
    if (request.raw.iterations == normalized_iterations &&
        request.raw.warmupIterations == normalized_warmup_iterations) {
        out << "\n    iterations: " << request.raw.iterations << " (warmup "
            << request.raw.warmupIterations << ")";
    } else {
        out << "\n    requested iterations: " << request.raw.iterations << " (warmup "
            << request.raw.warmupIterations << ")"
            << "\n    normalized iterations: " << normalized_iterations << " (warmup "
            << normalized_warmup_iterations << ")"
            << "\n    normalization: aligned for thread and large-block iteration distribution";
    }
    out << "\n    threads: " << request.raw.threads
        << "\n    pipeline depth: " << request.raw.pipelineDepth
        << "\n    consistency check: " << (request.raw.checkConsistency ? "enabled" : "disabled");
    if (request.hasFileOptions) {
        out << "\n  file-resource options:\n"
            << "    path: "
            << (request.file.path.empty() ? "<current working directory>" : request.file.path)
            << "\n    filenames: "
            << (request.file.filenames.empty() ? "<automatic>" : request.file.filenames)
            << "\n    files: " << request.file.numFiles
            << "\n    direct I/O: " << (request.file.direct ? "enabled" : "disabled");
    }
    out << "\n  plugin parameters:\n";
    for (const auto &key : sortedParameterKeys(request.pluginParameters)) {
        out << "    " << key << ": " << request.pluginParameters.at(key) << '\n';
    }
    if (request.raw.dryRun) {
        out << "Dry run: no worker was created and no allocation or transfer was attempted.\n";
    }
}

bool
isRawCommand(int argc, char *argv[]) {
    return argc > 1 && std::string_view(argv[1]) == "raw";
}

int
parseRawPosixCommand(int argc,
                     char *argv[],
                     const pluginMetadata &metadata,
                     rawPosixRequest &request,
                     bool &help_requested,
                     std::ostream &out,
                     std::ostream &err) {
    help_requested = false;
    request.pluginParameters = metadata.parameters;
    request.hasFileOptions = hasMemoryType(metadata, FILE_SEG);

    std::string total_buffer_size = std::to_string(request.raw.totalBufferSize);
    std::string start_block_size = std::to_string(request.raw.startBlockSize);
    std::string max_block_size = std::to_string(request.raw.maxBlockSize);
    std::vector<std::pair<std::string, std::string>> plugin_parameter_overrides;

    CLI::App app("NIXL data-transfer benchmark");
    app.require_subcommand(1);
    auto *raw = app.add_subcommand("raw", "Configure a low-level benchmark explicitly");
    raw->require_subcommand(1);
    auto *posix = raw->add_subcommand("posix", "Run the installed POSIX storage backend");
    posix->fallthrough();
    posix->footer("Raw benchmark options are documented by 'nixlbench raw --help' and may be used "
                  "before or after the posix subcommand.");

    raw->add_option("--operation", request.raw.operation, "Transfer direction: read or write")
        ->check(CLI::IsMember({"read", "write"}, CLI::ignore_case))
        ->group("Raw benchmark options");
    raw->add_option("--total-buffer-size",
                    total_buffer_size,
                    "Total buffer size using binary units (for example 4MiB or 4MB)")
        ->default_str(formatSize(request.raw.totalBufferSize))
        ->group("Raw benchmark options");
    raw->add_option("--start-block-size", start_block_size, "First block size in the sweep")
        ->default_str(formatSize(request.raw.startBlockSize))
        ->group("Raw benchmark options");
    raw->add_option("--max-block-size", max_block_size, "Last block size in the sweep")
        ->default_str(formatSize(request.raw.maxBlockSize))
        ->group("Raw benchmark options");
    raw->add_option(
           "--start-batch-size", request.raw.startBatchSize, "First batch size in the sweep")
        ->group("Raw benchmark options");
    raw->add_option("--max-batch-size", request.raw.maxBatchSize, "Last batch size in the sweep")
        ->group("Raw benchmark options");
    raw->add_option("--iterations", request.raw.iterations, "Timed iterations")
        ->group("Raw benchmark options");
    raw->add_option("--warmup-iterations", request.raw.warmupIterations, "Warmup iterations")
        ->group("Raw benchmark options");
    raw->add_option("--threads", request.raw.threads, "Benchmark worker threads")
        ->group("Raw benchmark options");
    raw->add_option("--pipeline-depth", request.raw.pipelineDepth, "Transfer requests in flight")
        ->group("Raw benchmark options");
    raw->add_flag("--check-consistency", request.raw.checkConsistency, "Validate transferred bytes")
        ->group("Raw benchmark options");
    raw->add_flag("--dry-run", request.raw.dryRun, "Print the resolved plan without executing")
        ->group("Raw benchmark options");

    if (request.hasFileOptions) {
        posix->add_option("--path", request.file.path, "Directory for automatically named files")
            ->group("FILE_SEG resource options");
        posix
            ->add_option(
                "--filenames", request.file.filenames, "Comma-separated explicit file names")
            ->group("FILE_SEG resource options");
        posix->add_option("--num-files", request.file.numFiles, "Number of backing files")
            ->group("FILE_SEG resource options");
        posix->add_flag("--direct", request.file.direct, "Use direct file opening")
            ->group("FILE_SEG resource options");
    }
    if (!metadata.parameters.empty()) {
        posix
            ->add_option("--plugin-param",
                         plugin_parameter_overrides,
                         pluginParameterDescription(metadata.parameters))
            ->check(CLI::IsMember(sortedParameterKeys(metadata.parameters))
                        .description("")
                        .application_index(0))
            ->type_name("KEY VALUE")
            ->group("Plugin initialization parameters");
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

    if (!hasMemoryType(metadata, DRAM_SEG)) {
        err << "Error: " << metadata.name << " plugin must advertise DRAM_SEG for local memory\n";
        return invalid_arguments_exit_code;
    }
    if (!hasMemoryType(metadata, FILE_SEG)) {
        err << "Error: " << metadata.name << " plugin must advertise FILE_SEG for backing files\n";
        return invalid_arguments_exit_code;
    }

    for (const auto &[key, value] : plugin_parameter_overrides) {
        request.pluginParameters[key] = value;
    }

    request.raw.operation = upper(request.raw.operation);
    std::vector<std::pair<const std::string *, size_t *>> sizes = {
        {&total_buffer_size, &request.raw.totalBufferSize},
        {&start_block_size, &request.raw.startBlockSize},
        {&max_block_size, &request.raw.maxBlockSize},
    };
    for (const auto &[text, destination] : sizes) {
        std::string size_error;
        const auto parsed = parseHumanSize(*text, size_error);
        if (!parsed) {
            err << "Error: invalid size '" << *text << "': " << size_error << '\n';
            return invalid_arguments_exit_code;
        }
        *destination = *parsed;
    }

    if (!validateRawOptions(request.raw, err)) {
        return invalid_arguments_exit_code;
    }
    if (request.hasFileOptions) {
        std::string file_error;
        if (!validateFileOptions(request.file, file_error)) {
            err << "Error: " << file_error << '\n';
            return invalid_arguments_exit_code;
        }
        if (request.file.numFiles > request.raw.threads ||
            request.raw.threads % request.file.numFiles != 0) {
            err << "Error: --num-files must divide --threads and cannot exceed it\n";
            return invalid_arguments_exit_code;
        }
    }
    return EXIT_SUCCESS;
}

int
parseRawCommand(int argc,
                char *argv[],
                const std::vector<pluginMetadata> &file_plugins,
                rawPosixRequest &request,
                bool &help_requested,
                std::ostream &out,
                std::ostream &err) {
    const auto metadata =
        std::find_if(file_plugins.begin(), file_plugins.end(), [](const pluginMetadata &candidate) {
            return candidate.name == XFERBENCH_BACKEND_POSIX;
        });
    if (metadata == file_plugins.end()) {
        err << "Error: " << XFERBENCH_BACKEND_POSIX
            << " plugin is not installed or does not advertise FILE_SEG\n";
        return EXIT_FAILURE;
    }
    return parseRawPosixCommand(argc, argv, *metadata, request, help_requested, out, err);
}

std::vector<std::string>
benchmarkFileArguments(const rawPosixRequest &request, const std::string &program_name) {
    const auto boolean = [](bool value) { return value ? "true" : "false"; };
    std::vector<std::string> arguments = {
        program_name,
        // Fixed values select the existing NIXL/POSIX runner.
        std::string("--worker_type=") + XFERBENCH_WORKER_NIXL,
        std::string("--backend=") + XFERBENCH_BACKEND_POSIX,
        std::string("--initiator_seg_type=") + XFERBENCH_SEG_TYPE_DRAM,
        // Backend-neutral raw benchmark configuration.
        "--op_type=" + request.raw.operation,
        "--check_consistency=" + std::string(boolean(request.raw.checkConsistency)),
        "--total_buffer_size=" + std::to_string(request.raw.totalBufferSize),
        "--start_block_size=" + std::to_string(request.raw.startBlockSize),
        "--max_block_size=" + std::to_string(request.raw.maxBlockSize),
        "--start_batch_size=" + std::to_string(request.raw.startBatchSize),
        "--max_batch_size=" + std::to_string(request.raw.maxBatchSize),
        "--num_iter=" + std::to_string(request.raw.iterations),
        "--warmup_iter=" + std::to_string(request.raw.warmupIterations),
        "--num_threads=" + std::to_string(request.raw.threads),
        "--pipeline_depth=" + std::to_string(request.raw.pipelineDepth)};
    if (request.hasFileOptions) {
        arguments.push_back("--filepath=" + request.file.path);
        arguments.push_back("--filenames=" + request.file.filenames);
        arguments.push_back("--num_files=" + std::to_string(request.file.numFiles));
        arguments.push_back("--storage_enable_direct=" + std::string(boolean(request.file.direct)));
    }
    return arguments;
}

rawCommandResult
prepareRawCommand(int argc, char *argv[], std::ostream &out, std::ostream &err) {
    std::string discovery_error;
    const auto file_plugins = discoverPluginsWithMemoryType(FILE_SEG, discovery_error);
    if (!file_plugins) {
        err << "Error: " << discovery_error << '\n';
        return {EXIT_FAILURE, false};
    }

    rawPosixRequest request;
    bool help_requested = false;
    const int parse_status =
        parseRawCommand(argc, argv, *file_plugins, request, help_requested, out, err);
    if (parse_status != EXIT_SUCCESS || help_requested) {
        return {parse_status, false};
    }

    auto arguments = benchmarkFileArguments(request, argv[0]);
    std::vector<char *> argument_pointers;
    argument_pointers.reserve(arguments.size());
    for (auto &argument : arguments) {
        argument_pointers.push_back(argument.data());
    }
    int legacy_argc = static_cast<int>(argument_pointers.size());
    char **legacy_argv = argument_pointers.data();
    if (xferBenchConfig::parseConfig(legacy_argc, legacy_argv) != EXIT_SUCCESS) {
        return {EXIT_FAILURE, false};
    }

    const auto metadata = std::find_if(
        file_plugins->begin(), file_plugins->end(), [](const pluginMetadata &candidate) {
            return candidate.name == XFERBENCH_BACKEND_POSIX;
        });
    printRawPosixPlan(
        request, *metadata, xferBenchConfig::num_iter, xferBenchConfig::warmup_iter, out);
    return {EXIT_SUCCESS, !request.raw.dryRun, std::move(request.pluginParameters)};
}

} // namespace nixlbench
