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

void load(const char *f, std::vector<uint64_t> &v) {
    auto a = cnpy::npy_load(f);
    auto p = a.data<uint64_t>();
    v.insert(v.end(), p, p + a.shape[0]);
}

void save(const char *f, const std::vector<uint64_t> &v) {
    auto shape = std::vector<size_t>{v.size()};
    cnpy::npy_save(f, v.data(), shape);
}

typedef std::tuple<uint64_t, uint64_t, uint64_t> cmnt_t;
typedef std::pair<uint64_t, uint64_t> pid_pair;

int main(int argc, char* *argv) {
    uint64_t T = atoi(argv[1]);
    bool stats_only = (argc > 2);

    std::vector<uint64_t> pid;
    std::vector<uint64_t> uid;
    std::vector<uint64_t> utc;
    #pragma omp parallel for
    for (int i = 0; i < 3; ++i) {
        switch (i) {
        case 0:
            load("pid.npy", pid);
            break;
        case 1:
            load("uid.npy", uid);
            break;
        case 2:
            load("utc.npy", utc);
            break;
        }
    }
    auto n_cmnts = pid.size();

    std::vector<cmnt_t> cmnts(n_cmnts);
    #pragma omp parallel for
    for (uint64_t i = 0; i < n_cmnts; ++i) {
        cmnts.at(i) = std::make_tuple(pid.at(i), uid.at(i), utc.at(i));
    }

    std::sort(cmnts.begin(), cmnts.end(),
              [](const cmnt_t &x, const cmnt_t &y) { return std::get<2>(x) < std::get<2>(y); });

    uint64_t progress = 0;
    std::vector<uint64_t> ns;
    std::vector<std::vector<pid_pair>> parallel_pid_pairs;
    if (stats_only) {
        ns.resize(n_cmnts);
    } else {
        parallel_pid_pairs.resize(n_cmnts);
    }
    #pragma omp parallel for
    for (uint64_t i = 0; i < n_cmnts; ++i) {
        auto pid_i = std::get<0>(cmnts.at(i));
        auto uid_i = std::get<1>(cmnts.at(i));
        auto utc_i = std::get<2>(cmnts.at(i));
        if (stats_only) {
            ns.at(i) = 0;
        }
        for (uint64_t j = i; j < n_cmnts; ++j) {
            auto pid_j = std::get<0>(cmnts.at(j));
            auto uid_j = std::get<1>(cmnts.at(j));
            auto utc_j = std::get<2>(cmnts.at(j));
            if (utc_i + T < utc_j) {
                break;
            }
            if (uid_j == uid_i) {
                if (stats_only) {
                    ++ns.at(i);
                } else {
                    parallel_pid_pairs.at(i).push_back(std::make_pair(pid_i, pid_j));
                    parallel_pid_pairs.at(i).push_back(std::make_pair(pid_j, pid_i));
                }
            }
        }

        uint64_t local_progress;
        #pragma omp atomic capture
        local_progress = ++progress;
        if (!(local_progress % 10000)) {
            std::cout << local_progress << '/' << n_cmnts << std::endl;
        }
    }

    if (stats_only) {
        uint64_t sum = 0;
        for (auto p = ns.begin(); p != ns.end(); ++p) {
            sum += *p;
        }
        std::cout << sum << std::endl;
    } else {
        std::vector<pid_pair> pid_pairs;
        for (auto p = parallel_pid_pairs.begin(); p != parallel_pid_pairs.end(); ++p) {
            pid_pairs.insert(pid_pairs.end(), p->begin(), p->end());
        }

        std::multiset<pid_pair> multiset(pid_pairs.begin(), pid_pairs.end());
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
        #pragma omp parallel for
        for (int i = 0; i < 3; ++i) {
            switch (i) {
            case 0:
                save("u.npy", u);
                break;
            case 1:
                save("v.npy", v);
                break;
            case 2:
                save("w.npy", w);
                break;
            }
        }
    }

    return 0;
}
