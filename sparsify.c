#include <stdio.h>

#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <vector>

#include <omp.h>

const uint64_t len_indptr =  3246684 + 1;
const uint64_t len_indices = 4040770;
const uint64_t threshold = 4;

uint64_t indptr[len_indptr];
uint64_t indices[len_indices];
std::vector<std::unordered_set<uint64_t>> rows(len_indptr - 1);

int main() {
    FILE *indptrf = fopen("indptr", "r");
    for (uint64_t i = 0; i < len_indptr; ++i) {
        fscanf(indptrf, "%lu", indptr + i);
    }

    FILE *indicesf = fopen("indices", "r");
    for (uint64_t i = 0; i < len_indices; ++i) {
        fscanf(indicesf, "%lu", indices + i);
    }

    /*
    std::cout << *std::max_element(indptr, indptr + len_indptr) << ' '
              << *std::min_element(indptr, indptr + len_indptr) << std::endl;
    std::cout << *std::max_element(indices, indices + len_indices) << ' '
              << *std::min_element(indices, indices + len_indices) << std::endl;
    */

    #pragma omp parallel for
    for (uint64_t i = 0; i < rows.size(); ++i) {
        auto m = indices + indptr[i];
        auto n = indices + indptr[i + 1];
        rows[i] = m < n ? std::unordered_set<uint64_t>(m, n) : std::unordered_set<uint64_t>();
    }

    std::vector<uint64_t> u;
    std::vector<uint64_t> v;
    omp_lock_t lock;
    omp_init_lock(&lock);
    #pragma omp parallel for
    for (uint64_t i = 0; i < rows.size(); ++i) {
        if (rows[i].size() < threshold) {
            continue;
        }
        std::vector<uint64_t> uu;
        std::vector<uint64_t> vv;
        for (uint64_t j = i + 1; j < rows.size(); ++j) {
            uint64_t k = 0;
            if (rows[j].size() < threshold) {
                continue;
            }
            for (const auto &x : (rows[i].size() < rows[j].size() ? rows[i] : rows[j])) {
                k += rows[i].count(x);
            }
            if (k >= threshold) {
                uu.push_back(i);
                vv.push_back(j);
            }
        }
        omp_set_lock(&lock);
        std::copy(uu.begin(), uu.end(), std::back_inserter(u));
        std::copy(vv.begin(), vv.end(), std::back_inserter(v));
        omp_unset_lock(&lock);
    }

    FILE *uf = fopen("u.sparsified", "w");
    FILE *vf = fopen("v.sparsified", "w");
    for (uint64_t i = 0; i < u.size(); ++i) {
        fprintf(uf, "%lu\n", u[i]);
        fprintf(vf, "%lu\n", v[i]);
    }

    return 0;
}
