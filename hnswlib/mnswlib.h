#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include "hnswalg.h"
#include <random>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <unordered_set>
#include <unordered_map>


namespace hnswlib {
template<typename dist_t>
class MarkovNSW : public HierarchicalNSW<dist_t> {
 public:
  MarkovNSW(SpaceInterface<dist_t> *s) : HierarchicalNSW<dist_t>(s) {}

  MarkovNSW(SpaceInterface<dist_t> *s,
      const std::string &location, bool nmslib = false, size_t max_elements=0) :
      HierarchicalNSW<dist_t>(s, location, nmslib, max_elements) {}

  MarkovNSW(SpaceInterface<dist_t> *s, size_t max_elements,
      size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100) :
      HierarchicalNSW<dist_t>(s, max_elements, M, ef_construction, random_seed) {}

  // call HierarchicalNSW::~HierarchicalNSW() automatically
  ~ MarkovNSW() = default;

  using PriorityQueue = std::priority_queue<
      std::pair<dist_t, tableint>,
      std::vector<std::pair<dist_t, tableint > >,
      typename HierarchicalNSW<dist_t>::CompareByFirst >;
  PriorityQueue searchBaseLayerSTMarkov(tableint ep_id, const void *data_point, size_t ef)  {
    VisitedList *vl = this->visited_list_pool_->getFreeVisitedList();
    vl_type *visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    PriorityQueue top_candidates;
    ProbPriorityQueue<std::pair<dist_t, tableint>, double> candidate_set;
    dist_t dist = this->fstdistfunc_(data_point, this->getDataByInternalId(ep_id), this->dist_func_param_);
    this->dist_calc++;

    top_candidates.emplace(dist, ep_id);
    candidate_set.emplace(std::pow(2.0, -dist), std::make_pair(-dist, ep_id));
    visited_array[ep_id] = visited_array_tag;
    dist_t lower_bound = dist;

    for (int i=0; i < ef; i++) {

      std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

      tableint current_node_id = current_node_pair.second;

      candidate_set.clear();

      int *data = (int *) (this->data_level0_memory_ + current_node_id * this->size_data_per_element_ + this->offsetLevel0_);
      int size = *data;
#ifdef USE_SSE
      _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
      _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
      _mm_prefetch(this->data_level0_memory_ + (*(data + 1)) * this->size_data_per_element_ + this->offsetData_, _MM_HINT_T0);
      _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

      for (int j = 1; j <= size; j++) {
        int candidate_id = *(data + j);
#ifdef USE_SSE
        _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
        _mm_prefetch(this->data_level0_memory_ + (*(data + j + 1)) * this->size_data_per_element_ + this->offsetData_,
                     _MM_HINT_T0);////////////
#endif
        // if (!(visited_array[candidate_id] == visited_array_tag))
        {

          visited_array[candidate_id] = visited_array_tag;

          char *currObj1 = (this->getDataByInternalId(candidate_id));
          dist_t dist = this->fstdistfunc_(data_point, currObj1, this->dist_func_param_);
          this->dist_calc++;
          candidate_set.emplace(std::pow(2.0, -dist), std::make_pair(-dist, candidate_id));

          if (!(visited_array[candidate_id] == visited_array_tag)) {
            if (top_candidates.top().first > dist || top_candidates.size() < ef) {
              top_candidates.emplace(dist, candidate_id);

              if (top_candidates.size() > ef) {
                top_candidates.pop();
              }
              lower_bound = top_candidates.top().first;
            }
          }
        }
      }
    }

    this->visited_list_pool_->releaseVisitedList(vl);
    return top_candidates;
  }


  std::priority_queue<std::pair<dist_t, labeltype >> searchKnn(const void *query_data, size_t k)  {
    tableint currObj = this->enterpoint_node_;
    dist_t curdist = this->fstdistfunc_(query_data,
        this->getDataByInternalId(this->enterpoint_node_), this->dist_func_param_);
    this->dist_calc++;

    for (int level = this->maxlevel_; level > 0; level--) {
      bool changed = true;
      while (changed) {
        changed = false;
        int *data;
        data = (int *) (this->linkLists_[currObj] + (level - 1) * this->size_links_per_element_);
        int size = *data;
        tableint *datal = (tableint *) (data + 1);
        for (int i = 0; i < size; i++) {
          tableint cand = datal[i];
          if (cand < 0 || cand > this->max_elements_)
            throw std::runtime_error("cand error");
          dist_t d = this->fstdistfunc_(query_data,
              this->getDataByInternalId(cand), this->dist_func_param_);
          this->dist_calc++;

          if (d < curdist) {
            curdist = d;
            currObj = cand;
            changed = true;
          }
        }
      }
    }

    PriorityQueue top_candidates = searchBaseLayerSTMarkov(
        currObj, query_data, std::max(this->ef_,k));
    std::priority_queue<std::pair<dist_t, labeltype >> results;
    while (top_candidates.size() > k) {
      top_candidates.pop();
    }
    while (top_candidates.size() > 0) {
      std::pair<dist_t, tableint> rez = top_candidates.top();
      results.push(std::pair<dist_t, labeltype>(rez.first, this->getExternalLabel(rez.second)));
      top_candidates.pop();
    }
    return results;
  };

}; // class MarkovNSW

} // namespace hnswlib
