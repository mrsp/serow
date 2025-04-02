#include "ThreadPool.hpp"

namespace serow {

ThreadPool::ThreadPool(size_t num_threads) : stop_(false), active_jobs_(0) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back(&ThreadPool::worker, this);
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (std::thread& worker : workers_) {
        worker.join();
    }
}

void ThreadPool::worker() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

            if (stop_ && tasks_.empty()) {
                return;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
        }
        task();
    }
}

}  // namespace serow
