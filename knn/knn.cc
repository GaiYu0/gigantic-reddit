#include <assert.h>
#include <stdlib.h>

#include <fstream>
#include <utility>
#include <vector>

#include <omp.h>

void savetxt(const char *f, std::vector<uint64_t> &a) {
    std::ofstream s(f);
    for (auto p = a.begin(); p != a.end(); ++p) {
        s << *p << std::endl;
    }
}

void loadtxt(const char *f, std::vector<uint64_t> &a) {
    std::ifstream s(f);
    for (auto p = a.begin(); p != a.end(); ++p) {
        s >> *p;
    }
}

typedef counter_t std::pair<uint64_t, uint64_t>;
typedef multiset_t std::vector<std::pair<uint64_t, uint64_t>>;
void merge(multiset_t &lhs, multiset_t &rhs, multiset_t &ret) {
    auto p = lhs.begin();
    auto q = rhs.begin();
    while (p != lhs.end() && q != rhs.end()) {
        if (p->first < q->first) {
            ret.push_back(*p);
            ++p;
        } else if (p->first > q->first) {
            ret.push_back(*q);
            ++q;
        } else {
            ret.push_back(std::make_pair(p->first, p->second + q->second));
            ++p;
            ++q;
        }
    }
    if (p != lhs.end()) {
        ret.insert(ret.end(), p, lhs.end());
    } else if (q != rhs.end()) {
        ret.insert(ret.end(), q, rhs.end());
    }
    return;
}

int main(int argc, char* *argv) {
    auto n_posts = atoi(argv[1]);
    auto n_cmnts = atoi(argv[2]);
    auto n_users = atoi(argv[3]);
    auto degree = atoi(argv[4]);

    std::vector<uint64_t> p2u_indptr(n_posts);
    std::vector<uint64_t> p2u_indices(n_cmnts);
    std::vector<uint64_t> u2p_indptr(n_users);
    std::vector<uint64_t> u2p_indices(n_cmnts);
    #pragma omp parallel for
    for (int i = 0; i < 4; ++i) {
        switch (i) {
        case 0:
            loadtxt("p2u-indptr", p2u_indptr);
            break;
        case 1:
            loadtxt("p2u-indices", p2u_indices);
            break;
        case 2:
            loadtxt("u2p-indptr", u2p_indptr);
            break;
        case 3:
            loadtxt("u2p-indices", u2p_indices);
            break;
        }
    }

    std::vector<std::vector<uint64_t>> knn(n_posts);
    #pragma omp parallel for
    for (uint64_t i = 0; i < n_posts; ++i) {
        auto begin = p2u_indptr[i];
        auto end = p2u_indptr[i + 1];
        assert(begin < end);
        auto &prev = multiset_t();
        prev.reserve(u2p_indptr[begin + 1] - u2p_indptr[begin]);
        for (uint64_t k = u2p_indptr[begin]; k < u2p_indptr[begin + 1]; ++k) {
            prev.push_back(std::make_pair(u2p_indices[k], 1));
        }
        for (uint64_t j = begin + 1; j < end; ++j) {
            auto next = multiset_t();
            next.reserve(u2p_indptr[begin + 1] - u2p_indptr[begin]);
            for (uint64_t k = u2p_indptr[j]; k < u2p_indptr[j + 1]; ++k) {
                next.push_back(std::make_pair(u2p_indices[k], 1));
            }
            auto ret = multiset_t();
            // ret.reserve(prev.size() + next.size());
            merge(prev, next, ret);
            prev = ret;
        }
        std::sort(prev.begin(), prev.end(),
                  [](counter_t lhs, counter_t rhs) { lhs->second < rhs->second; });
        for (uint64_t j = 0; j < degree && j < prev.size(); ++j) {
            knn[i].push_back(prev.rbegin()[j].first);
        }
    }

    std::vector<uint64_t> src, dst;
    auto m = degree * n_posts;
    src.reserve(m);
    dst.reserve(m);
    for (auto p = knn.begin(); p != knn.end(); ++p) {
        for (auto q = p->begin(); q != p->end(); ++q) {
            src.push_back(p - knn.begin());
            dst.push_back(*q);
        }
    }
    savetxt("src", src);
    savetxt("dst", dst);

    return 0;
}
