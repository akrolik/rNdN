#pragma once

#include <vector>

#include "CUDA/Allocator.h"

namespace CUDA {

template<typename T>
using Vector = std::vector<T, Allocator<T>>;
// using Vector = std::vector<T, std::allocator<T>>;

}
