#pragma once

#include <vector>

#include "CUDA/Allocator.h"

namespace CUDA {

template<typename T, class A = PinnedAllocator<T>>
using Vector = std::vector<T, A>;

template<typename T>
using PinnedVector = Vector<T, PinnedAllocator<T>>;

template<typename T>
using MappedVector = Vector<T, MappedAllocator<T>>;

}
