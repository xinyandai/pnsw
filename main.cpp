#include <iostream>
#include <fstream>
#include <string>
#include <queue>
#include <chrono>
#include <unordered_set>

#include "assert.h"
#include "rss.hpp"
#include "recall.hpp"
#include "hnswlib/hnswlib.h"

using namespace std;
using namespace hnswlib;

const int defaultEfConstruction = 1024;
const int defaultEfSearch = 128;
const int defaultM = 32;
const int defaultTopK = 10;

template <typename DistType>
void outDegree(HierarchicalNSW<DistType > *appr_alg, float *x, int vecsize, int vecdim) {
    std::vector<float> element_norms;
    element_norms.reserve(vecsize);
    for (int i = 0; i < vecsize; ++i) {
        float line_norm = 0;
        for (int j = 0; j < vecdim; ++j) {
            float ele = x[i * vecdim + j];
            line_norm += ele * ele;
        }
        line_norm = sqrt(line_norm);
        element_norms.push_back(line_norm);
    }

    std::vector<int> external_count(appr_alg->max_elements_);
    int *temp_data = NULL;
    int degree_count = 0;
    for (int i = 0; i < appr_alg->max_elements_; ++i) {
        temp_data = (int *)(appr_alg->data_level0_memory_ + i * appr_alg->size_data_per_element_);
        int degree = *temp_data;
        for (int j = 1; j <= degree; ++j) {
            external_count[appr_alg->getExternalLabel(*(temp_data + j))]++;
        }
        degree_count += degree;
        // std::cout << "norm : " << norm << ", degrees : " << degree  << std::endl;
        // std::cout << norm << ", " << degree  << std::endl;
    }
    std::cout << "avg. degree = " << (float)degree_count / appr_alg->max_elements_  << std::endl;
    for (int i = 0; i < appr_alg->max_elements_; ++i) {
        float norm = element_norms[appr_alg->getExternalLabel(i)];
        std::cout << norm << ", " << external_count[appr_alg->getExternalLabel(i)] << std::endl;
    }
    return ;
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}

template <typename DataType>
void loadData(DataType*& data, size_t& dimension, size_t & cardinality, const std::string& inputPath) {
    std::ifstream fin(inputPath.c_str(), std::ios::binary | std::ios::ate);
    if (!fin) {
        std::cout << "cannot open file " << inputPath << std::endl;
        exit(1);
    }

    size_t fileSize = fin.tellg();
    fin.seekg(0, fin.beg);
    if (fileSize == 0) {
        std::cout << "file size is 0 " << inputPath << std::endl;
        exit(1);
    }

    int dim;
    fin.read(reinterpret_cast<char*>(&dim), sizeof(int));
    dimension = (size_t)dim;
    size_t bytesPerRecord = dimension * sizeof(DataType) + 4;
    if (fileSize % bytesPerRecord != 0) {
        std::cout << "File not aligned" << std::endl;
        exit(1);
    }
    cardinality = fileSize / bytesPerRecord;
    data = new DataType[cardinality * dimension];
    fin.read((char*)data, sizeof(DataType) * dimension);

    for (int i = 1; i < cardinality; ++i) {
        fin.read((char*)&dim, 4);
        assert(dim == dimension);
        fin.read((char*)(data + i * dimension), sizeof(DataType) * dimension);
    }
    fin.close();
}

using namespace std;
using namespace hnswlib;


void run_hnsw();

int main(int argc, char** argv) {
    run_hnsw();
}

void run_hnsw() {
    using DistType = float;
    using DistSpace = L2Space;
    char *path_q = "/research/jcheng2/xinyan/jie-data/imagenet/imagenet_query.fvecs";
    char *path_x = "/research/jcheng2/xinyan/jie-data/imagenet/imagenet_base.fvecs";
    char *path_gt = "/research/jcheng2/xinyan/jie-data/imagenet/imagenet_top20_product_groundtruth.ivecs";
    char path_index[1024];

    float * x;
    float * q;
    int * gt;
    size_t qsize;
    size_t vecdim;
    size_t vecsize;
    size_t gt_k;
    std::cout << "loading data" << std::endl;
    loadData(x, vecdim, vecsize, path_x);
    std::cout << "loading query" << std::endl;
    loadData(q, vecdim, qsize, path_q);
    std::cout << "loading ground truth" << std::endl;
    loadData(gt, gt_k, qsize, path_gt);

    int efConstruction = 128;
    int M = 32;

    sprintf(path_index, "%s_ef_%d_M_%d.bin", "imagenet", efConstruction, M);

    DistSpace l2space(vecdim);

    HierarchicalNSW<DistType > *appr_alg;
    if (exists_test(path_index)) {
        cout << "Loading index from " << path_index << ":\n";
        appr_alg = new HierarchicalNSW<DistType >(&l2space, path_index, false);
        cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    } else {
        cout << "Building index:\n";
        appr_alg = new HierarchicalNSW<DistType >(&l2space, vecsize, M, efConstruction);

        appr_alg->addPoint((void *) (x), (size_t) 0);
        int j1 = 0;
        StopW stopw = StopW();
        StopW stopw_full = StopW();
        size_t report_every = 100000;
#pragma omp parallel for
        for (int i = 1; i < vecsize; i++) {
#pragma omp critical
            {
                j1++;
                if (j1 % report_every == 0) {
                    cout << j1 / (0.01 * vecsize) << " %, "
                         << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips " << " Mem: "
                         << getCurrentRSS() / 1000000 << " Mb \n";
                    stopw.reset();
                }
            }
            appr_alg->addPoint( (void *) (x + vecdim * i), (size_t) i);


        }
        cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\n";
        appr_alg->saveIndex(path_index);
    }

    outDegree(appr_alg, x, vecsize, vecdim);

    delete[] x;
    delete[] q;

    vector<std::priority_queue<std::pair<float, labeltype >>> answers;
    size_t k = 10;
    cout << "Parsing gt:\n";
    get_gt(gt, qsize, l2space, answers, k, gt_k);
    cout << "Loaded gt\n";
    for (int i = 0; i < 1; i++)
        test_vs_recall(q, qsize, *appr_alg, vecdim, answers, k);
    cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";

    delete[] gt;
}