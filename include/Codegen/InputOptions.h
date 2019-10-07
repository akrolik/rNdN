#pragma once

#include <string>

namespace Codegen {

struct InputOptions
{
	constexpr static unsigned long DynamicSize = 0;

	unsigned long ActiveThreads = DynamicSize;
	unsigned long ActiveBlocks = DynamicSize;

	std::string ToString() const
	{
		std::string output;
		output += "Active threads: " + ((ActiveThreads == DynamicSize) ? "<dynamic>" : std::to_string(ActiveThreads)) + "\n";
		output += "Active blocks: " + ((ActiveBlocks == DynamicSize) ? "<dynamic>" : std::to_string(ActiveBlocks));
		return output;
	}
};

}
