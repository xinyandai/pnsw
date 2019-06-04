//
// Created by xinyan on 2/6/2019.
//

#pragma once
#include <cmath>
#include <random>
#include <utility>
#include <vector>

/***
 * @tparam T type of generated random number, float or double
 * @param range
 * @return return a uniform random number between [.0, range]
 */
template <typename T>
T uniformRealDistribution(T range) {
  static std::default_random_engine generator;
  std::uniform_real_distribution<T> distribution(0.0, range);
  return distribution(generator);
}

template<class T, typename ProbType = float>
class ProbPriorityQueue {
 private:
  vector<std::pair<ProbType, T > > queue_;
  ProbType prob_norm_;
 public:
  ProbPriorityQueue() : queue_(), prob_norm_(0.0) {}

  /***
   * randomly pop one element according to its probability priority
   * @return
   */
  T pop() {
    ProbType r = uniformRealDistribution<ProbType >(prob_norm_);
    for (auto iter = queue_.begin(); iter!=queue_.end(); iter++) {
      if (r <= iter->first || iter+1 == queue_.end()) {
        prob_norm_ -= iter->first;
        T re = iter->second;
        queue_.erase(iter, iter+1);
        return re;
      } else {
        r -= iter->first;
      }
    }
    throw std::runtime_error("probability overflow");
  }
  std::pair<ProbType, T >& top() {
    int best_i = 0;
    for (int i=1; i<queue_.size(); i++) {
      if (queue_[i].first > queue_[best_i].first) {
        best_i = i;
      }
    }
    return queue_[best_i];

  }
  /*
  std::pair<ProbType, T > pop() {
    int best_i = 0;
    for (int i=1; i<queue_.size(); i++) {
      if (queue_[i].first > queue_[best_i].first) {
        best_i = i;
      }
    }
    std::pair<ProbType, T > re = queue_[best_i];
    queue_.erase(queue_.begin() + best_i);
    return re;

  }
  */
  void emplace(ProbType prob, const T &obj) {
    prob_norm_ += prob;
    queue_.push_back(std::make_pair(prob, obj));
  }

  bool empty() {
    return queue_.size() == 0;
  }
};
