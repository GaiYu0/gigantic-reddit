#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <iterator>
#include <utility>
#include <vector>

template<typename T> void savetxt(const char *f, std::vector<T> &a) {
    std::ofstream s(f);
    for (auto p = a.begin(); p != a.end(); ++p) {
        s << *p << std::endl;
    }
}

template<typename T> void loadtxt(const char *f, std::vector<T> &a) {
    std::ifstream s(f);
    std::copy(std::istream_iterator<T>(s),
              std::istream_iterator<T>(),
              std::back_inserter(a));
}

int main(int argc, char* *argv) {
    auto K = atoi(argv[1]);

    std::vector<double> lhs_data;
    std::vector<uint64_t> lhs_indptr;
    std::vector<uint64_t> lhs_indices;
    std::vector<double> rhs_data;
    std::vector<uint64_t> rhs_indptr;
    std::vector<uint64_t> rhs_indices;

    #pragma omp parallel for
    for (int i = 0; i < 6; ++i) {
        switch (i) {
        case 0:
            loadtxt(argv[2], lhs_data);
            break;
        case 1:
            loadtxt(argv[3], lhs_indptr);
            break;
        case 2:
            loadtxt(argv[4], lhs_indices);
            break;
        case 3:
            loadtxt(argv[5], rhs_data);
            break;
        case 4:
            loadtxt(argv[6], rhs_indptr);
            break;
        case 5:
            loadtxt(argv[7], rhs_indices);
            break;
        }
    }

    uint64_t progress = 0;
    std::vector<std::vector<std::pair<uint64_t, double>>> rows(lhs_indptr.size() - 1);
    #pragma omp parallel for
    for (uint64_t i = 0; i < lhs_indptr.size() - 1; ++i) {
        std::map<uint64_t, double> map;
        for (uint64_t j = lhs_indptr[i]; j < lhs_indptr[i + 1]; ++j)  {
            for (uint64_t k = rhs_indptr[lhs_indices[j]];
                 k < rhs_indptr[lhs_indices[j] + 1]; ++k) {
                map[rhs_indices[k]] += lhs_data[j] * rhs_data[k];
            }
        }
        std::vector<std::pair<uint64_t, double>> vector(map.begin(), map.end());
        auto compare = [](const std::pair<uint64_t, double> &lhs, const std::pair<uint64_t, double> &rhs) { return lhs.second < rhs.second; };
        std::stable_sort(vector.begin(), vector.end(), compare);
        rows[i].insert(rows[i].end(), vector.rbegin(), vector.rbegin() + K);
        uint64_t local_progress;
        #pragma omp atomic capture
        local_progress = ++progress;
        if (!(local_progress % 10000)) {
            std::cout << local_progress << '/' << lhs_indptr.size() << std::endl;
        }
    }

    std::vector<uint64_t> src, dst;
    std::vector<double> dat;
    auto len = K * lhs_indptr.size();
    src.reserve(len);
    dst.reserve(len);
    dat.reserve(len);
    for (uint64_t i = 0; i < rows.size(); ++i) {
        src.insert(src.end(), rows[i].size(), i);
        for (uint64_t j = 0; j < rows.size(); ++j) {
            dst.push_back(rows[i][j].first);
            dat.push_back(rows[i][j].second);
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < 3; ++i) {
        switch (i) {
        case 0:
            savetxt("src", src);
            break;
        case 1:
            savetxt("dst", dst);
            break;
        case 2:
            savetxt("dat", dat);
            break;
        }
    }

    return 0;
}
