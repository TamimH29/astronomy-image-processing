#pragma once
#include <algorithm>
#include <cstddef>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace astro_cpu {
class Serial {
public:
    std::size_t size() const { return 1; }
    template<class F> void run(std::size_t n, F&& task) {
        task(std::size_t{0}, std::size_t{0}, n);
    }
};

// The calling thread is worker 0. Workers and their scratch storage are reused
// through every stage of this image; no process-wide mutable pool is shared.
class Workers {
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable work_, done_;
    std::function<void(std::size_t, std::size_t, std::size_t)> task_;
    std::size_t size_, count_ = 0, generation_ = 0, remaining_ = 0;
    bool stopping_ = false;
    std::exception_ptr failure_;

    void invoke(std::size_t worker) {
        try {
            const auto base = count_ / size_, extra = count_ % size_;
            const auto begin = worker * base + std::min(worker, extra);
            task_(worker, begin, begin + base + (worker < extra));
        } catch (...) {
            std::lock_guard lock(mutex_);
            if (!failure_) failure_ = std::current_exception();
        }
    }
    void stop() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        work_.notify_all();
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
    }
public:
    explicit Workers(std::size_t size) : size_(size) {
        try {
            for (std::size_t i = 1; i < size_; ++i) {
                workers_.emplace_back([this, i] {
                    std::size_t seen = 0;
                    std::unique_lock lock(mutex_);
                    for (;;) {
                        work_.wait(lock, [&] { return stopping_ || generation_ != seen; });
                        if (stopping_) return;
                        seen = generation_;
                        lock.unlock();
                        invoke(i);
                        lock.lock();
                        if (--remaining_ == 0) done_.notify_one();
                    }
                });
            }
        } catch (...) {
            stop();
            throw;
        }
    }
    ~Workers() { stop(); }
    std::size_t size() const { return size_; }
    template<class F> void run(std::size_t count, F&& task) {
        if (size_ == 1) {
            task(std::size_t{0}, std::size_t{0}, count);
            return;
        }
        {
            std::lock_guard lock(mutex_);
            task_ = std::forward<F>(task);
            count_ = count;
            failure_ = nullptr;
            remaining_ = size_ - 1;
            ++generation_;
        }
        work_.notify_all();
        invoke(0);
        std::unique_lock lock(mutex_);
        done_.wait(lock, [&] { return remaining_ == 0; });
        task_ = {};
        if (failure_) std::rethrow_exception(failure_);
    }
};

} // namespace astro_cpu
