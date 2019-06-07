#include <stdlib.h>
#include <fstream>
#include <omp.h>

void savetxt(const char *f, std::vector<uint64_t> &a) {
    std::fstream s(f, s.out);
    for (auto p = a.begin(); p != a.end(); ++p) {
        s << *p << std::endl;
    }
}

void loadtxt(const char *f, std::vector<uint64_t> &a) {
    std::fstream s(f, s.in);
    for (auto p = a.begin(); p != a.end(); ++p) {
        s >> *p;
    }
}

int main(int argc, char* *argv) {
    uint64_t m = atoi(argv[1]);
    uint64_t T = atoi(argv[2]);

    std::vector<uint64_t> pid(m);
    std::vector<uint64_t> uid(m);
    std::vector<uint64_t> utc(m);
    #pragma omp parallel for
    for (int i = 0; i < 3; ++i) {
        switch (i) {
        case 0:
            loadtxt("pid", sid);
            break;
        case 1:
            loadtxt("uid", uid);
            break;
        case 2:
            loadtxt("utc", utc);
            break;
        }
    }

    typedef std::tuple<uint64_t, uint64_t, uint64_t> cmnt_t;
    std::vector<cmnt_t> cmnts(m);
    #pragma omp parallel for
    for (uint64_t i = 0; i < m; ++i) {
        cmnts.at(i) = std::make_tuple(pid.at[i], uid.at(i), utc.at(i));
    }

    std::sort(cmnts.begin(), cmnts.end(),
              [](cmnt_t c, cmnt_t d) { return std::get<3>(c) < std::get<3>(d); });

    typedef std::pair<uint64_t, uint64_t> pid_pair;
    std::vector<std::vector<pid_pair>> parallel_pid_pairs(m);
    #pragma omp parallel for
    for (uint64_t i = 0; i < m; ++i) {
        auto pid_i = std::get<2>(cmnts.at(i));
        auto uid_i = std::get<2>(cmnts.at(i));
        auto utc_i = std::get<3>(cmnts.at(i));
        for (uint64_t j = i; j < m; ++j) {
            auto pid_j = std::get<2>(cmnts.at(j));
            auto uid_j = std::get<2>(cmnts.at(j));
            auto utc_j = std::get<3>(cmnts.at(j));
            if (utc_j < utc_i + T) {
                break;
            }
            if (uid_j == uid_i) {
                parallel_pid_pairs.at(i).push_back(std::make_pair(pid_i, pid_j));
                parallel_pid_pairs.at(i).push_back(std::make_pair(pid_j, pid_i));
            }
        }
    }

    std::vector<pid_pair> pid_pairs;
    for (const auto &v : parallel_pid_pairs) {
        pid_pairs.insert(pid_pairs.end(), v.begin(), v.end());
    }

    auto pid_pair_less = [](pid_pair p, pid_pair q) {
        return p.first == q.first ? p.second < q.second : p.first < q.first; };
    typedef std::multiset<pid_pair, pid_pair_less> pid_pair_multiset;
    pid_pair_multiset multiset(pid_pairs.begin(), pid_pairs.end());
    std::vector<pid_t> u, v, w;
    u.reserve(multiset.size());
    v.reserve(multiset.size());
    w.reserve(multiset.size());
    for (const auto pq : multiset) {
        u.push_back(pq.first);
        v.push_back(pq.second);
        w.push_back(multiset.count(pq);
    }

    save_txt("u", u);
    save_txt("v", v);
    save_txt("w", w);

    return 0;
}
