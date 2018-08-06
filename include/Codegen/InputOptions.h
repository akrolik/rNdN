#pragma once

#include <string>

namespace Codegen {

struct InputOptions
{
	constexpr static unsigned long DynamicSize = 0;

	unsigned long InputSize = DynamicSize;

	std::string ToString() const
	{
		std::string output;
		output += "Input size: " + ((InputSize == DynamicSize) ? "dynamic" : std::to_string(InputSize));
		return output;
	}
};

}
