#include "lya/backend.hpp"
#include <stdexcept>

namespace lya {
std::unique_ptr<Backend> make_cuda_backend(const Header&,const Spectrum&,const std::vector<Source>&,const BackendSettings&) {
    throw std::runtime_error("CUDA was disabled at build time; rebuild with -DLYA_ENABLE_CUDA=ON");
}
std::uint64_t cuda_free_memory(int) {
    throw std::runtime_error("CUDA was disabled at build time; use --backend cpu only for tiny validation fixtures");
}
} // namespace lya
