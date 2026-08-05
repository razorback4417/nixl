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

    constexpr int kInvalidArgumentsExitCode = 2;

    std::string
    upper(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
            return static_cast<char>(std::toupper(ch));
        });
        return value;
    }

    bool
    validateRawOptions(const RawOptions &raw, std::ostream &err) {
        const auto fail = [&](const std::string &message) {
            err << "Error: " << message << '\n';
            return false;
        };

        if (raw.operation != XFERBENCH_OP_READ && raw.operation != XFERBENCH_OP_WRITE) {
            return fail("--operation must be read or write");
        }
        if (raw.threads < 1 || raw.iterations < 1 || raw.warmup_iterations < 0 ||
            raw.pipeline_depth < 1) {
            return fail(
                "threads, iterations, and pipeline depth must be positive; warmup may be zero");
        }
        if (raw.start_block_size == 0 || raw.max_block_size < raw.start_block_size) {
            return fail("block sizes must be positive and max must be at least start");
        }
        if (raw.start_batch_size == 0 || raw.max_batch_size < raw.start_batch_size) {
            return fail("batch sizes must be positive and max must be at least start");
        }
        return true;
    }

} // namespace

void
printRawPosixPlan(const RawPosixRequest &request,
                  const PluginMetadata &metadata,
                  int normalized_iterations,
                  int normalized_warmup_iterations,
                  std::ostream &out) {
    out << "Resolved NIXLBench plan\n"
        << "  command: raw posix\n"
        << "  backend: " << metadata.name << "\n"
        << "  memory types: ";
    auto memory_types = metadata.memory_types;
    std::sort(memory_types.begin(), memory_types.end());
    for (size_t i = 0; i < memory_types.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << nixlEnumStrings::memTypeStr(memory_types[i]);
    }
    out << "\n  benchmark options:\n"
        << "    operation: " << request.raw.operation
        << "\n    total buffer: " << formatSize(request.raw.total_buffer_size)
        << "\n    block sizes: " << formatSize(request.raw.start_block_size) << " .. "
        << formatSize(request.raw.max_block_size)
        << "\n    batch sizes: " << request.raw.start_batch_size << " .. "
        << request.raw.max_batch_size;
    if (request.raw.iterations == normalized_iterations &&
        request.raw.warmup_iterations == normalized_warmup_iterations) {
        out << "\n    iterations: " << request.raw.iterations << " (warmup "
            << request.raw.warmup_iterations << ")";
    } else {
        out << "\n    requested iterations: " << request.raw.iterations << " (warmup "
            << request.raw.warmup_iterations << ")"
            << "\n    normalized iterations: " << normalized_iterations << " (warmup "
            << normalized_warmup_iterations << ")"
            << "\n    normalization: aligned for thread and large-block iteration distribution";
    }
    out << "\n    threads: " << request.raw.threads
        << "\n    pipeline depth: " << request.raw.pipeline_depth
        << "\n    consistency check: " << (request.raw.check_consistency ? "enabled" : "disabled");
    if (request.has_file_options) {
        out << "\n  file-resource options:\n"
            << "    path: "
            << (request.file.path.empty() ? "<current working directory>" : request.file.path)
            << "\n    filenames: "
            << (request.file.filenames.empty() ? "<automatic>" : request.file.filenames)
            << "\n    files: " << request.file.num_files
            << "\n    direct I/O: " << (request.file.direct ? "enabled" : "disabled");
    }
    out << "\n  plugin parameters:\n";
    for (const auto &key : sortedParameterKeys(request.plugin_parameters)) {
        out << "    " << key << ": " << request.plugin_parameters.at(key) << '\n';
    }
    if (request.raw.dry_run) {
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
                     const PluginMetadata &metadata,
                     RawPosixRequest &request,
                     bool &help_requested,
                     std::ostream &out,
                     std::ostream &err) {
    help_requested = false;
    request.plugin_parameters = metadata.parameters;
    request.has_file_options = hasMemoryType(metadata, FILE_SEG);

    std::string total_buffer_size = std::to_string(request.raw.total_buffer_size);
    std::string start_block_size = std::to_string(request.raw.start_block_size);
    std::string max_block_size = std::to_string(request.raw.max_block_size);
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
        ->default_str(formatSize(request.raw.total_buffer_size))
        ->group("Raw benchmark options");
    raw->add_option("--start-block-size", start_block_size, "First block size in the sweep")
        ->default_str(formatSize(request.raw.start_block_size))
        ->group("Raw benchmark options");
    raw->add_option("--max-block-size", max_block_size, "Last block size in the sweep")
        ->default_str(formatSize(request.raw.max_block_size))
        ->group("Raw benchmark options");
    raw->add_option(
           "--start-batch-size", request.raw.start_batch_size, "First batch size in the sweep")
        ->group("Raw benchmark options");
    raw->add_option("--max-batch-size", request.raw.max_batch_size, "Last batch size in the sweep")
        ->group("Raw benchmark options");
    raw->add_option("--iterations", request.raw.iterations, "Timed iterations")
        ->group("Raw benchmark options");
    raw->add_option("--warmup-iterations", request.raw.warmup_iterations, "Warmup iterations")
        ->group("Raw benchmark options");
    raw->add_option("--threads", request.raw.threads, "Benchmark worker threads")
        ->group("Raw benchmark options");
    raw->add_option("--pipeline-depth", request.raw.pipeline_depth, "Transfer requests in flight")
        ->group("Raw benchmark options");
    raw->add_flag(
           "--check-consistency", request.raw.check_consistency, "Validate transferred bytes")
        ->group("Raw benchmark options");
    raw->add_flag("--dry-run", request.raw.dry_run, "Print the resolved plan without executing")
        ->group("Raw benchmark options");

    if (request.has_file_options) {
        posix->add_option("--path", request.file.path, "Directory for automatically named files")
            ->group("FILE_SEG resource options");
        posix
            ->add_option(
                "--filenames", request.file.filenames, "Comma-separated explicit file names")
            ->group("FILE_SEG resource options");
        posix->add_option("--num-files", request.file.num_files, "Number of backing files")
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
        return kInvalidArgumentsExitCode;
    }
    if (!hasMemoryType(metadata, FILE_SEG)) {
        err << "Error: " << metadata.name << " plugin must advertise FILE_SEG for backing files\n";
        return kInvalidArgumentsExitCode;
    }

    for (const auto &[key, value] : plugin_parameter_overrides) {
        request.plugin_parameters[key] = value;
    }

    request.raw.operation = upper(request.raw.operation);
    std::vector<std::pair<const std::string *, size_t *>> sizes = {
        {&total_buffer_size, &request.raw.total_buffer_size},
        {&start_block_size, &request.raw.start_block_size},
        {&max_block_size, &request.raw.max_block_size},
    };
    for (const auto &[text, destination] : sizes) {
        std::string size_error;
        const auto parsed = parseHumanSize(*text, size_error);
        if (!parsed) {
            err << "Error: invalid size '" << *text << "': " << size_error << '\n';
            return kInvalidArgumentsExitCode;
        }
        *destination = *parsed;
    }

    if (!validateRawOptions(request.raw, err)) {
        return kInvalidArgumentsExitCode;
    }
    if (request.has_file_options) {
        std::string file_error;
        if (!validateFileOptions(request.file, file_error)) {
            err << "Error: " << file_error << '\n';
            return kInvalidArgumentsExitCode;
        }
        if (request.file.num_files > request.raw.threads ||
            request.raw.threads % request.file.num_files != 0) {
            err << "Error: --num-files must divide --threads and cannot exceed it\n";
            return kInvalidArgumentsExitCode;
        }
    }
    return EXIT_SUCCESS;
}

int
parseRawCommand(int argc,
                char *argv[],
                const std::vector<PluginMetadata> &file_plugins,
                RawPosixRequest &request,
                bool &help_requested,
                std::ostream &out,
                std::ostream &err) {
    const auto metadata =
        std::find_if(file_plugins.begin(), file_plugins.end(), [](const PluginMetadata &candidate) {
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
benchmarkFileArguments(const RawPosixRequest &request, const std::string &program_name) {
    const auto boolean = [](bool value) { return value ? "true" : "false"; };
    std::vector<std::string> arguments = {
        program_name,
        // Fixed values select the existing NIXL/POSIX runner.
        std::string("--worker_type=") + XFERBENCH_WORKER_NIXL,
        std::string("--backend=") + XFERBENCH_BACKEND_POSIX,
        std::string("--initiator_seg_type=") + XFERBENCH_SEG_TYPE_DRAM,
        // Backend-neutral raw benchmark configuration.
        "--op_type=" + request.raw.operation,
        "--check_consistency=" + std::string(boolean(request.raw.check_consistency)),
        "--total_buffer_size=" + std::to_string(request.raw.total_buffer_size),
        "--start_block_size=" + std::to_string(request.raw.start_block_size),
        "--max_block_size=" + std::to_string(request.raw.max_block_size),
        "--start_batch_size=" + std::to_string(request.raw.start_batch_size),
        "--max_batch_size=" + std::to_string(request.raw.max_batch_size),
        "--num_iter=" + std::to_string(request.raw.iterations),
        "--warmup_iter=" + std::to_string(request.raw.warmup_iterations),
        "--num_threads=" + std::to_string(request.raw.threads),
        "--pipeline_depth=" + std::to_string(request.raw.pipeline_depth)};
    if (request.has_file_options) {
        arguments.push_back("--filepath=" + request.file.path);
        arguments.push_back("--filenames=" + request.file.filenames);
        arguments.push_back("--num_files=" + std::to_string(request.file.num_files));
        arguments.push_back("--storage_enable_direct=" + std::string(boolean(request.file.direct)));
    }
    return arguments;
}

RawCommandResult
prepareRawCommand(int argc, char *argv[], std::ostream &out, std::ostream &err) {
    std::string discovery_error;
    const auto file_plugins = discoverPluginsWithMemoryType(FILE_SEG, discovery_error);
    if (!file_plugins) {
        err << "Error: " << discovery_error << '\n';
        return {EXIT_FAILURE, false};
    }

    RawPosixRequest request;
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
        file_plugins->begin(), file_plugins->end(), [](const PluginMetadata &candidate) {
            return candidate.name == XFERBENCH_BACKEND_POSIX;
        });
    printRawPosixPlan(
        request, *metadata, xferBenchConfig::num_iter, xferBenchConfig::warmup_iter, out);
    return {EXIT_SUCCESS, !request.raw.dry_run, std::move(request.plugin_parameters)};
}

} // namespace nixlbench
