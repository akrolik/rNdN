#pragma once

#include <cstddef>
#include <cstdint>

namespace SASS {

namespace Maxwell {
	constexpr unsigned int MIN_COMPUTE_CAPABILITY = 50;
	constexpr unsigned int MAX_COMPUTE_CAPABILITY = 63;

	constexpr std::size_t CBANK_PARAM_OFFSET = 0x140;

	static bool IsSupported(unsigned int computeCapability)
	{
		return (computeCapability >= MIN_COMPUTE_CAPABILITY && computeCapability <= MAX_COMPUTE_CAPABILITY);
	}
}

namespace Volta {
	constexpr unsigned int MIN_COMPUTE_CAPABILITY = 70;
	constexpr unsigned int MAX_COMPUTE_CAPABILITY = 86;

	constexpr std::size_t CBANK_PARAM_OFFSET = 0x160;
	constexpr std::size_t MAX_SSY_STACK_DEPTH = 15;

	static bool IsSupported(unsigned int computeCapability)
	{
		return (computeCapability >= MIN_COMPUTE_CAPABILITY && computeCapability <= MAX_COMPUTE_CAPABILITY);
	}
}

constexpr std::size_t SSY_STACK_SIZE = 0x10;
constexpr std::uint32_t WARP_SIZE = 32u;
constexpr std::uint32_t MAX_BARRIERS = 16u;

}
