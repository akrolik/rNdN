#pragma once

#include <string>

namespace Frontend {
namespace Codegen {

struct TargetOptions
{
	std::string ComputeCapability = "sm_61";
	std::uint32_t MaxBlockSize = 512;
	std::uint32_t WarpSize = 32;
	std::uint32_t SharedMemorySize = 48000;

	std::string ToString() const
	{
		std::string output;
		output += "Compute capability: " + ComputeCapability + "\n";
		output += "Max block size: " + std::to_string(MaxBlockSize) + "\n";
		output += "Warp size: " + std::to_string(WarpSize) + "\n";
		output += "Shared memory size: " + std::to_string(SharedMemorySize);
		return output;
	}
};

}
}
