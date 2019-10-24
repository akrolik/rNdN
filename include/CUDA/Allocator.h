#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

#include "CUDA/Utils.h"

namespace CUDA {

template <class T>
struct Allocator
{
	typedef T value_type;

	Allocator() = default;

	template <class U>
	constexpr Allocator(const Allocator<U>&) noexcept {}

	[[nodiscard]] T *allocate(std::size_t n)
	{
		if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
		{
			throw std::bad_alloc();
		}

		T* p;
		checkRuntimeError(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocDefault));
		return p;
	}

	void deallocate(T *p, std::size_t) noexcept
	{
		cudaFreeHost(p);
	}
};

template <class T, class U>
bool operator==(const Allocator<T>&, const Allocator<U>&) { return true; }

template <class T, class U>
bool operator!=(const Allocator<T>&, const Allocator<U>&) { return false; }

}
