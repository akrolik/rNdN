#pragma once

#include <string>

namespace Codegen {

struct TargetOptions
{
	std::string ComputeCapability = "sm_61";
	unsigned int MaxBlockSize;

	std::string ToString() const
	{
		std::string output;
		output += "Compute capability: " + ComputeCapability + "\n";
		output += "Max block size: " + std::to_string(MaxBlockSize) + "\n";
		return output;
	}
};

}
