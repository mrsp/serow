#ifndef QUEUE_ROS_H
#define QUEUE_ROS_H
#include <queue>
#include <thread>
#include <mutex>
#include <vector>
#include <condition_variable>

 template <typename T>
class Queue
{
 public:

  T pop() 
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty())
    {
      cond_.wait(mlock);
    }
    auto val = queue_.front();
    queue_.pop();
    return val;
  }

  void pop(T& item)
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    while (queue_.empty())
    {
      cond_.wait(mlock);
    }
    item = queue_.front();
    queue_.pop();
  }

  int size() 
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    return queue_.size();
  }
  
  bool empty() 
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    return queue_.empty();
  }
  
  void push(const T& item)
  {
    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.push(item);
    mlock.unlock();
    cond_.notify_one();
  }
  
  Queue()=default;
  Queue(const Queue&) = delete;            // disable copying
  Queue& operator=(const Queue&) = delete; // disable assignment
  
 private:
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
};
#endif