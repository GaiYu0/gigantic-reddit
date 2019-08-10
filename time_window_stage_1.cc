#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <set>
#include <tuple>
#include <utility>
#include <omp.h>
#include "cnpy.h"

typedef std::tuple<uint64_t, uint64_t, uint64_t> cmnt_t;
typedef std::pair<uint64_t, uint64_t> pid_pair;
typedef std::multiset<pid_pair> pid_pair_multiset;

int main(int argc, char* *argv) {
    uint64_t m = atoi(argv[1]);
    uint64_t T = atoi(argv[2]);
    bool stats_only = (argc > 3);

    std::vector<uint64_t> pid(m);
    std::vector<uint64_t> uid(m);
    std::vector<uint64_t> utc(m);
    #pragma omp parallel for
    for (int i = 0; i < 3; ++i) {
        switch (i) {
        case 0:
            loadtxt("pid", pid);
            break;
        case 1:
            loadtxt("uid", uid);
            break;
        case 2:
            loadtxt("utc", utc);
            break;
        }
    }

    std::vector<cmnt_t> cmnts(m);
    #pragma omp parallel for
    for (uint64_t i = 0; i < m; ++i) {
        cmnts.at(i) = std::make_tuple(pid.at(i), uid.at(i), utc.at(i));
    }

    std::sort(cmnts.begin(), cmnts.end(),
              [](const cmnt_t &c, const cmnt_t &d) { return std::get<2>(c) < std::get<2>(d); });

    std::vector<uint64_t> parallel_n_pid_pairs;
    std::vector<std::vector<pid_pair>> parallel_pid_pairs;
    if (stats_only) {
        parallel_n_pid_pairs.resize(m);
    } else {
        parallel_pid_pairs.resize(m);
    }
    #pragma omp parallel for
    for (uint64_t i = 0; i < m; ++i) {
        auto pid_i = std::get<0>(cmnts.at(i));
        auto uid_i = std::get<1>(cmnts.at(i));
        auto utc_i = std::get<2>(cmnts.at(i));
        if (stats_only) {
            parallel_n_pid_pairs.at(i) = 0;
        }
        for (uint64_t j = i; j < m; ++j) {
            auto pid_j = std::get<0>(cmnts.at(j));
            auto uid_j = std::get<1>(cmnts.at(j));
            auto utc_j = std::get<2>(cmnts.at(j));
            if (utc_i + T < utc_j) {
                break;
            }
            if (uid_j == uid_i) {
                if (stats_only) {
                    ++parallel_n_pid_pairs.at(i);
                } else {
                    parallel_pid_pairs.at(i).push_back(std::make_pair(pid_i, pid_j));
                    parallel_pid_pairs.at(i).push_back(std::make_pair(pid_j, pid_i));
                }
            }
        }
    }

    if (stats_only) {
        uint64_t sum = 0;
        for (auto p = parallel_n_pid_pairs.begin(); p != parallel_n_pid_pairs.end(); ++p) {
            sum += *p;
        }
        std::cout << sum << std::endl;
    } else {
        std::vector<pid_pair> pid_pairs;
        for (auto p = parallel_pid_pairs.begin(); p != parallel_pid_pairs.end(); ++p) {
            pid_pairs.insert(pid_pairs.end(), p->begin(), p->end());
        }

        pid_pair_multiset multiset(pid_pairs.begin(), pid_pairs.end());
        std::vector<uint64_t> u, v, w;
        u.reserve(multiset.size());
        v.reserve(multiset.size());
        w.reserve(multiset.size());
        for (auto p = multiset.begin(); p != multiset.end(); p = multiset.upper_bound(*p)) {
            u.push_back(p->first);
            v.push_back(p->second);
            w.push_back(multiset.count(*p));
        }

        std::cout << u.size() << std::endl;
        savetxt("u", u);
        savetxt("v", v);
        savetxt("w", w);
    }

    return 0;
}
