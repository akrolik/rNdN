#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace Codegen {

struct InputOptions
{
	const Analysis::Shape *ThreadGeometry = nullptr;

	constexpr static std::uint32_t DynamicSize = 0;
	std::uint32_t ListCellThreads = DynamicSize;

	std::unordered_map<const HorseIR::SymbolTable::Symbol *, const Analysis::Shape *> ParameterShapes;
	std::vector<const Analysis::Shape *> ReturnShapes;

	std::string ToString() const
	{
		std::string output;
		output += "Thread geometry: " + Analysis::ShapeUtils::ShapeString(ThreadGeometry) + "\n";
		if (Analysis::ShapeUtils::IsShape<Analysis::ListShape>(ThreadGeometry))
		{
			output += "List cell threads: " + ((ListCellThreads == DynamicSize) ? "<dynamic>" : std::to_string(ListCellThreads)) + "\n";
		}
		output += "Parameter shapes: ";
		if (ParameterShapes.size() > 0)
		{
			bool first = true;
			for (const auto& [parameter, shape] : ParameterShapes)
			{
				if (!first)
				{
					output += ", ";
				}
				first = false;
				output += parameter->name + " = " + Analysis::ShapeUtils::ShapeString(shape);
			}
		}
		else
		{
			output += "-";
		}
		output += "\nReturn shapes: ";
		if (ReturnShapes.size() > 0)
		{
			bool first = true;
			for (const auto& shape : ReturnShapes)
			{
				if (!first)
				{
					output += ", ";
				}
				first = false;
				output += Analysis::ShapeUtils::ShapeString(shape);
			}
		}
		else
		{
			output += "-";
		}
		return output;
	}
};

}
