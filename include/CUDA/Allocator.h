#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>

#include "CUDA/Utils.h"

namespace CUDA {

template <class T>
struct PinnedAllocator
{
	typedef T value_type;

	PinnedAllocator() = default;

	template <class U>
	constexpr PinnedAllocator(const PinnedAllocator<U>&) noexcept {}

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
		checkRuntimeError(cudaFreeHost(p));
	}
};

template <class T, class U>
bool operator==(const PinnedAllocator<T>&, const PinnedAllocator<U>&) { return true; }

template <class T, class U>
bool operator!=(const PinnedAllocator<T>&, const PinnedAllocator<U>&) { return false; }

template <class T>
struct MappedAllocator
{
	typedef T value_type;

	MappedAllocator() = default;

	template <class U>
	constexpr MappedAllocator(const MappedAllocator<U>&) noexcept {}

	[[nodiscard]] T *allocate(std::size_t n)
	{
		if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
		{
			throw std::bad_alloc();
		}

		T* p;
		checkRuntimeError(cudaHostAlloc(&p, n * sizeof(T), cudaHostAllocMapped));
		return p;
	}

	void deallocate(T *p, std::size_t) noexcept
	{
		checkRuntimeError(cudaFreeHost(p));
	}
};

template <class T, class U>
bool operator==(const MappedAllocator<T>&, const MappedAllocator<U>&) { return true; }

template <class T, class U>
bool operator!=(const MappedAllocator<T>&, const MappedAllocator<U>&) { return false; }

}
