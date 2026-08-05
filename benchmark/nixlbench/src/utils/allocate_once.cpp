/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utils/allocate_once.h"

#include "utils/utils.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <ostream>
#include <set>
#include <sys/stat.h>
#include <unordered_set>
#include <unistd.h>
#include <utility>

namespace nixlbench {
namespace {

    constexpr size_t kInitializationChunk = 1024 * 1024;

    class FileHandle {
    public:
        explicit FileHandle(int fd) : fd_(fd) {}

        ~FileHandle() {
            if (fd_ >= 0) {
                ::close(fd_);
            }
        }

        FileHandle(const FileHandle &) = delete;
        FileHandle &
        operator=(const FileHandle &) = delete;

    private:
        int fd_;
    };

    bool
    initializeManagedFile(int fd, uint64_t size, size_t alignment, bool direct, std::ostream &err) {
        if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
            err << "Failed to resize managed file: " << strerror(errno) << '\n';
            return false;
        }

        const size_t chunk_size = std::min<size_t>(kInitializationChunk, size);
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

std::optional<std::vector<ThreadFileRegion>>
allocateOnceThreadRegions(const AllocateOnceRequest &request, std::string &error) {
    if (request.files.empty() || request.threads < 1 || request.block_size == 0) {
        error = "files, threads, and block size must be configured";
        return std::nullopt;
    }
    const uint64_t slots_per_file = request.file_size / request.block_size;
    std::vector<size_t> threads_per_file(request.files.size(), 0);
    for (int thread = 0; thread < request.threads; ++thread) {
        ++threads_per_file[static_cast<size_t>(thread) % request.files.size()];
    }

    std::vector<size_t> file_thread_index(request.files.size(), 0);
    std::vector<ThreadFileRegion> regions;
    regions.reserve(static_cast<size_t>(request.threads));
    for (int thread = 0; thread < request.threads; ++thread) {
        const size_t file_index = static_cast<size_t>(thread) % request.files.size();
        const size_t ordinal = file_thread_index[file_index]++;
        const size_t count = threads_per_file[file_index];
        const uint64_t first = slots_per_file * ordinal / count;
        const uint64_t end = slots_per_file * (ordinal + 1) / count;
        if (end - first < request.batch_size) {
            error = "each thread partition must contain at least --batch-size blocks";
            return std::nullopt;
        }
        regions.push_back({file_index, first, end - first});
    }
    return regions;
}

OffsetSequence::OffsetSequence(ThreadFileRegion region,
                               size_t batch_size,
                               OffsetMode mode,
                               uint64_t seed)
    : region_(region),
      batch_size_(batch_size),
      mode_(mode),
      next_sequential_slot_(region.first_slot),
      random_(seed) {}

std::vector<uint64_t>
OffsetSequence::next() {
    std::vector<uint64_t> result;
    result.reserve(batch_size_);
    if (mode_ == OffsetMode::Sequential) {
        for (size_t index = 0; index < batch_size_; ++index) {
            result.push_back(next_sequential_slot_);
            const uint64_t relative = next_sequential_slot_ - region_.first_slot;
            next_sequential_slot_ = region_.first_slot + (relative + 1) % region_.slot_count;
        }
        return result;
    }

    std::unordered_set<uint64_t> selected;
    const uint64_t start = region_.slot_count - batch_size_;
    for (uint64_t candidate = start; candidate < region_.slot_count; ++candidate) {
        std::uniform_int_distribution<uint64_t> distribution(0, candidate);
        const uint64_t random_slot = distribution(random_);
        const uint64_t chosen =
            selected.find(random_slot) == selected.end() ? random_slot : candidate;
        selected.insert(chosen);
        result.push_back(region_.first_slot + chosen);
    }
    return result;
}

bool
prepareAllocateOnceFiles(const AllocateOnceRequest &request, std::ostream &err) {
    const long page_size_value = sysconf(_SC_PAGESIZE);
    if (page_size_value <= 0) {
        err << "Could not determine system page size\n";
        return false;
    }
    const size_t alignment = static_cast<size_t>(page_size_value);
    std::set<std::pair<dev_t, ino_t>> identities;

    for (const auto &path : request.files) {
        int flags = (request.managed_files || request.operation == NIXL_WRITE ? O_RDWR : O_RDONLY) |
            O_LARGEFILE;
        if (request.managed_files) {
            flags |= O_CREAT | O_NOFOLLOW;
        }
        if (request.direct) {
            flags |= O_DIRECT;
        }

        const int fd = open(path.c_str(), flags, 0644);
        if (fd < 0) {
            err << "Failed to open " << path << ": " << strerror(errno) << '\n';
            return false;
        }
        FileHandle file(fd);

        struct stat info{};
        if (fstat(fd, &info) != 0) {
            err << "Failed to inspect " << path << ": " << strerror(errno) << '\n';
            return false;
        }
        if (request.managed_files && !S_ISREG(info.st_mode)) {
            err << "Managed path must be a regular file: " << path << '\n';
            return false;
        }
        if (!identities.emplace(info.st_dev, info.st_ino).second) {
            err << "Backing files must refer to distinct files: " << path << '\n';
            return false;
        }
        if (request.managed_files) {
            if (static_cast<uint64_t>(info.st_size) != request.file_size &&
                !initializeManagedFile(fd, request.file_size, alignment, request.direct, err)) {
                return false;
            }
        } else if (static_cast<uint64_t>(info.st_size) < request.file_size) {
            err << "Explicit file " << path << " is smaller than --file-size\n";
            return false;
        }
    }
    return true;
}

} // namespace nixlbench
