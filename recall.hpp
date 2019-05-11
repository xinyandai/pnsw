//
// Created by xinyan on 10/5/2019.
//

#pragma once
#ifndef HNSW_LIB_RECALL_HPP
#define HNSW_LIB_RECALL_HPP

#include <vector>
#include "hnswlib/hnswalg.h"
#include "hnswlib/hnswalg.h"

using namespace hnswlib;
using namespace std;

class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    long getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }

};

template <typename DistType, class Space>
void
get_gt(int *gt,
        size_t qsize,
        Space &l2space,
        vector<std::priority_queue<std::pair<DistType, labeltype >>> &answers,
        size_t k,
        size_t gt_k) {

    (vector<std::priority_queue<std::pair<DistType, labeltype >>>(qsize)).swap(answers);

    cout << qsize << "\n";
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, gt[gt_k * i + j]);
        }
    }
}

template <typename DataType, typename DistType=float>
float
test_approx(DataType *massQ, size_t qsize, HierarchicalNSW<DistType > &appr_alg, size_t vecdim,
            vector<std::priority_queue<std::pair<DistType, labeltype >>> &answers, size_t k) {
    size_t correct = 0;
    size_t total = 0;
    //uncomment to test in parallel mode:
    //#pragma omp parallel for
    for (int i = 0; i < qsize; i++) {

        std::priority_queue<std::pair<DistType, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k);
        std::priority_queue<std::pair<DistType, labeltype >> gt(answers[i]);
        unordered_set<labeltype> g;
        total += gt.size();

        while (gt.size()) {
            g.insert(gt.top().second);
            gt.pop();
        }

        while (result.size()) {
            if (g.find(result.top().second) != g.end()) {
                correct++;
            }
            result.pop();
        }

    }
    return 1.0f * correct / total;
}

template <typename DataType, typename DistType=float>
void
test_vs_recall(DataType *massQ, size_t qsize, HierarchicalNSW<DistType > &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<DistType, labeltype >>> &answers, size_t k) {
    vector<size_t> efs;// = { 10,10,10,10,10 };


    for (int i = 10; i < 100; i+=10) {
        efs.push_back(i);
    }
    for (int i = 100; i < 300; i += 20) {
        efs.push_back(i);
    }
    for (int i = 300; i < 1000; i += 100) {
        efs.push_back(i);
    }
    for (int i = 2000; i < 20000; i += 2000) {
        efs.push_back(i);
    }

    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        StopW stopw = StopW();

        float recall = test_approx(massQ, qsize, appr_alg, vecdim, answers, k);
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;

        cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\n";
        if (recall > 1.0) {
            cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
}

#endif //HNSW_LIB_RECALL_HPP
