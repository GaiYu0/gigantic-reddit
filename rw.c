#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

#include <omp.h>

const uint64_t n_authors = 1000000;
const uint64_t n_comments = 1000000;
const uint64_t n_submissions = 1000000;
const uint64_t n_samples = 1000000;

uint64_t a2s_indptr[n_authors];
uint64_t a2s_indices[n_comments];
uint64_t s2a_indptr[n_authors];
uint64_t s2a_indices[n_comments];
std::vector<std::unordered_set<uint64_t>> author_sets(n_submissions);
std::vector<float> norms(n_submissions);
std::vector<uint64_t> uu(n_samples);
std::vector<uint64_t> vv(n_samples);
std::vector<bool> accepted(n_samples);

void loadtxt(const char *f, uint64_t len, uint64_t *a) {
    FILE *p = fopen(f, "r");
    for (uint64_t i = 0; i < len; ++i) {
        fscanf(p, "%lu", a + i);
    }
}

int main() {
    loadtxt("a2s_indptr", n_authors, a2s_indptr);
    loadtxt("a2s_indices", n_authors, a2s_indices);
    loadtxt("s2a_indptr", n_authors, s2a_indptr);
    loadtxt("s2a_indices", n_authors, s2a_indices);

    #pragma omp parallel for
    for (uint64_t i = 0; i < n_submissions; ++i) {
        auto p = s2a_indices + s2a_indptr[i];
        auto q = s2a_indices + s2a_indptr[i + 1];
        author_sets[i] = p < q ? std::unordered_set<uint64_t>(p, q) : std::unordered_set<uint64_t>();
    }

    #pragma omp parallel for
    for (uint64_t i = 0; i < n_submissions; ++i) {
        norms[i] = 1 / sqrt(static_cast<float>(author_sets[i].size()));
    }

    std::vector<uint64_t> range(n_samples);
    std::iota(range.begin(), range.end(), 0);
    std::random_shuffle(range.begin(), range.end());
    for (uint64_t i = 0; i < n_samples; ++i) {
        auto u = range[i];
        std::random_device rd;
        auto engine = std::mt19937(rd);
        std::uniform_int_distribution<> author_distribution(s2a_indptr[u], s2a_indptr[u + 1]);
        auto a = s2a_indices[author_distribution(engine)];
        std::uniform_int_distribution<> submission_distribution(s2a_indptr[a], s2a_indptr[a + 1]);
        auto v = a2s_indices[submission_distribution(engine)];
        uu[i] = u;
        vv[i] = v;
    }

    for (uint64_t i = 0; i < n_samples; ++i) {
        auto u = uu[i];
        auto v = vv[i];
        uint64_t less, more;
        if (author_sets[u].size() < author_sets[v].size()) {
            less = u;
            more = v;
        } else {
            less = v;
            more = u;
        }
        uint64_t n = 0;
    }

    return 0;
}
