/**
 * Copyright (C) Stylianos Piperakis, Ownage Dynamics L.P.
 * Serow is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, version 3.
 *
 * Serow is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Serow. If not,
 * see <https://www.gnu.org/licenses/>.
 **/
#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace serow {

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
    ~ThreadPool();

    // Add a job to the threadpool
    template <typename F, typename... Args>
    auto addJob(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type> {
        using return_type = typename std::invoke_result<F, Args...>::type;
        using task_type = std::packaged_task<return_type()>;

        auto task =
            std::make_shared<task_type>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (stop_) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }

            if (tasks_.size() >= max_queue_size_) {
                tasks_.pop();  // drop oldest task (backpressure when queue is full)
            }

            tasks_.emplace([task, this]() {
                try {
                    active_jobs_++;
                    (*task)();
                } catch (...) {
                    // Ensure we decrement even if task throws
                    active_jobs_--;
                    throw;  // Re-throw the exception
                }
                active_jobs_--;
            });
        }
        condition_.notify_one();
        return res;
    }

    // Check if any jobs are currently running
    bool isRunning() const {
        return active_jobs_ > 0;
    }

private:
    // Worker thread function
    void worker();

    // Thread pool data
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    // Synchronization
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_;
    std::atomic<size_t> active_jobs_;
    size_t max_queue_size_{1000};
};

}  // namespace serow
