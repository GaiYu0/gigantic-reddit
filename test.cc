#include <iostream>
#include "cnpy.h"

int main(int argc, char* *argv) {
    cnpy::NpyArray a = cnpy::npy_load(argv[1]);
    uint64_t *p = a.data<uint64_t>();
    std::cout << a.shape.size() << ' ' << a.shape[0] << std::endl;
    std::cout << *p << std::endl;
    return 0;
}
