#include <assert.h>
#include <math.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_set>
#include <vector>

#include <boost/unordered_set.hpp>
#include <boost/functional/hash.hpp>
#include <omp.h>

typedef std::pair<uint64_t, uint64_t> edge_t;

const uint64_t n_authors = 4240972 - 1;
const uint64_t n_comments = 29753437;
const uint64_t n_submissions = 27166161 - 1;
const uint64_t n_samples = 1e9;
const float threshold = 0.001;

std::vector<uint64_t> a2s_indptr(n_authors + 1);
std::vector<uint64_t> a2s_indices(n_comments);
std::vector<uint64_t> s2a_indptr(n_submissions + 1);
std::vector<uint64_t> s2a_indices(n_comments);
std::vector<std::unordered_set<uint64_t>> a_sets(n_submissions);
std::vector<float> normalizers(n_submissions);
std::vector<uint64_t> uu(n_samples);
std::vector<uint64_t> vv(n_samples);
std::vector<bool> accepted(n_samples);

void savetxt(const char *f, uint64_t len, uint64_t *a) {
    FILE *p = fopen(f, "w");
    for (uint64_t i = 0; i < len; ++i) {
        fprintf(p, "%lu\n", a[i]);
    }
}

void loadtxt(const char *f, std::vector<uint64_t> &a) {
    std::fstream s(f, s.in);
    for (uint64_t i = 0; i < a.size(); ++i) {
        s >> a[i];
    }
}

int main() {
    #pragma omp parallel for
    for (int i = 0; i < 4; ++i) {
        switch (i) {
        case 0:
            loadtxt("a2s-indptr", a2s_indptr);
            break;
        case 1:
            loadtxt("a2s-indices", a2s_indices);
            break;
        case 2:
            loadtxt("s2a-indptr", s2a_indptr);
            break;
        case 3:
            loadtxt("s2a-indices", s2a_indices);
            break;
        }
    }

    std::cout << __LINE__ << std::endl;
    #pragma omp parallel for
    for (uint64_t i = 0; i < n_submissions; ++i) {
        auto p = s2a_indices.begin() + s2a_indptr[i];
        auto q = s2a_indices.begin() + s2a_indptr[i + 1];
        a_sets[i] = p < q ? std::unordered_set<uint64_t>(p, q) : std::unordered_set<uint64_t>();
    }

    std::cout << __LINE__ << std::endl;
    #pragma omp parallel for
    for (uint64_t i = 0; i < n_submissions; ++i) {
        normalizers[i] = 1 / sqrt(static_cast<float>(a_sets[i].size()));
    }

    std::cout << *std::max_element(normalizers.begin(), normalizers.end()) << ' '
              << *std::min_element(normalizers.begin(), normalizers.end()) << std::endl;

    std::cout << __LINE__ << std::endl;
    #pragma omp parallel for
    for (uint64_t i = 0; i < n_samples; ++i) {
        std::random_device rd;
        std::uniform_int_distribution<> u_distr(0, n_submissions - 1);
        auto u = u_distr(rd);
        std::uniform_int_distribution<> a_distr(s2a_indptr.at(u), s2a_indptr.at(u + 1) - 1);
        auto a = s2a_indices.at(a_distr(rd));
        std::uniform_int_distribution<> v_distr(a2s_indptr.at(a), a2s_indptr.at(a + 1) - 1);
        auto v = a2s_indices.at(v_distr(rd));
        uu.at(i) = u;
        vv.at(i) = v;

        uint64_t less, more;
        if (a_sets.at(u).size() < a_sets.at(v).size()) {
            less = u;
            more = v;
        } else {
            less = v;
            more = u;
        }
        uint64_t n = 0;
        for (const auto &a : a_sets.at(less)) {
            n += a_sets.at(more).count(a);
        }
        float cos = n * normalizers.at(u) * normalizers.at(v);
        accepted.at(i) = (cos >= threshold);
    }

    std::cout << __LINE__ << std::endl;
    boost::unordered_set<edge_t> edges;
    for (uint64_t i = 0; i < n_samples; ++i) {
        if (accepted[i]) {
            edges.insert(std::make_pair(uu[i], vv[i]));
        }
    }

    std::cout << __LINE__ << std::endl;
    std::vector<uint64_t> src;
    std::vector<uint64_t> dst;
    src.reserve(edges.size());
    dst.reserve(edges.size());
    for (const auto &uv : edges) {
        src.push_back(uv.first);
        dst.push_back(uv.second);
    }

    std::cout << "src.size()" << ' ' << src.size() << std::endl;
    std::cout << "dst.size()" << ' ' << dst.size() << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < 2; ++i) {
        switch (i) {
        case 0:
            savetxt("src.sparsified", src.size(), src.data());
            break;
        case 1:
            savetxt("dst.sparsified", dst.size(), dst.data());
            break;
        }
    }

    return 0;
}
