#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace Codegen {

struct InputOptions
{
	constexpr static unsigned long DynamicSize = 0;

	unsigned long ActiveThreads = DynamicSize;
	unsigned long ActiveBlocks = DynamicSize;

	enum class IndexKind {
		Scalar,  // data[0]
		Vector,  // data[thread id]
		List,    // data[cell id][local thread id]
		Cell     // data[cell id]
	};

	static std::string IndexKindString(IndexKind kind)
	{
		switch (kind)
		{
			case IndexKind::Scalar:
				return "Scalar";
			case IndexKind::Vector:
				return "Vector";
			case IndexKind::List:
				return "List";
			case IndexKind::Cell:
				return "Cell";
		}
		return "<unknown>";
	}

	std::unordered_map<const HorseIR::Parameter *, IndexKind> ParameterIndexKinds;
	std::vector<IndexKind> ReturnIndexKinds;

	std::string ToString() const
	{
		std::string output;
		output += "Active threads: " + ((ActiveThreads == DynamicSize) ? "<dynamic>" : std::to_string(ActiveThreads)) + "\n";
		output += "Active blocks: " + ((ActiveBlocks == DynamicSize) ? "<dynamic>" : std::to_string(ActiveBlocks)) + "\n";
		output += "Parameter indexes: ";
		bool first = true;
		for (const auto& param : ParameterIndexKinds)
		{
			if (!first)
			{
				output += ", ";
			}
			first = false;
			output += HorseIR::PrettyPrinter::PrettyString(param.first) + " = " + IndexKindString(param.second);
		}
		output += "\nReturn indexes: ";
		first = true;
		for (const auto& ret : ReturnIndexKinds)
		{
			if (!first)
			{
				output += ", ";
			}
			first = false;
			output += IndexKindString(ret);
		}
		return output;
	}
};

}
