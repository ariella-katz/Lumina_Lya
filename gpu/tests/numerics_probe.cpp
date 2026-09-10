#include "lya/numerics.hpp"
#include <iomanip>
#include <iostream>
#include <string>

// A small stdin/stdout interface for independent SciPy and CUDA comparisons.
int main() {
    using namespace lya::math;
    std::cout << std::setprecision(17);
    std::string operation;
    while (std::cin >> operation) {
        double x, hi, drift, a, weight;
        int second;
        if (operation == "dawson") { std::cin >> x; std::cout << dawson(x); }
        else if (operation == "hypergeo") { std::cin >> x; std::cout << hypergeo(x); }
        else if (operation == "h") {
            std::cin >> x >> a >> second; std::cout << voigt_h(x, a, second);
        } else if (operation == "integral") {
            std::cin >> x >> hi >> a >> second;
            std::cout << voigt_integral(x, hi, a, second);
        } else if (operation == "band") {
            std::cin >> x >> hi >> drift >> a >> second;
            std::cout << band_voigt_integral(x, hi, drift, a, second);
        } else if (operation == "delta") {
            std::cin >> x >> drift >> weight; std::cout << delta_cell_tau(x, drift, weight);
        } else if (operation == "band_delta") {
            std::cin >> x >> hi >> drift >> weight;
            std::cout << band_delta_tau(x, hi, drift, weight);
        } else {
            std::cerr << "Unknown probe operation: " << operation << '\n'; return 2;
        }
        if (!std::cin) return 2;
        std::cout << '\n';
    }
}
