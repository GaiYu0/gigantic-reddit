#include <algorithm>
#include <fstream>
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
            loadtxt("lhs-data", lhs_data);
            break;
        case 1:
            loadtxt("lhs-indptr", lhs_indptr);
            break;
        case 2:
            loadtxt("lhs-indices", lhs_indices);
            break;
        case 3:
            loadtxt("rhs-data", rhs_data);
            break;
        case 4:
            loadtxt("rhs-indptr", rhs_indptr);
            break;
        case 5:
            loadtxt("rhs-indices", rhs_indices);
            break;
        }
    }

    std::vector<std::vector<std::pair<uint64_t, double>>> rows;
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
