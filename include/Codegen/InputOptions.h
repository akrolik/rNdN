#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "Analysis/DataObject/DataObject.h"
#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace Codegen {

struct InputOptions
{
	// Prefix sum ordered thread groups

	bool InOrderBlocks = false;

	// Thread geometry and list cell count

	const Analysis::Shape *ThreadGeometry = nullptr;

	bool IsVectorGeometry() const { return Analysis::ShapeUtils::IsShape<Analysis::VectorShape>(ThreadGeometry); }
	bool IsListGeometry() const { return Analysis::ShapeUtils::IsShape<Analysis::ListShape>(ThreadGeometry); }

	constexpr static std::uint32_t DynamicSize = 0;
	std::uint32_t ListCellThreads = DynamicSize;

	// Parameter data

	std::unordered_map<const HorseIR::SymbolTable::Symbol *, const HorseIR::Parameter *> Parameters;

	std::unordered_map<const HorseIR::Parameter *, const Analysis::Shape *> ParameterShapes;

	std::unordered_map<const HorseIR::Parameter *, const Analysis::DataObject *> ParameterObjects;
	std::unordered_map<const Analysis::DataObject *, const HorseIR::Parameter *> ParameterObjectMap;

	// Declaration data

	std::unordered_map<const HorseIR::SymbolTable::Symbol *, const HorseIR::VariableDeclaration *> Declarations;
	std::unordered_map<const HorseIR::VariableDeclaration *, const Analysis::Shape *> DeclarationShapes;

	// Return data

	std::vector<const Analysis::Shape *> ReturnShapes;
	std::vector<const Analysis::Shape *> ReturnWriteShapes;
	std::vector<const Analysis::DataObject *> ReturnObjects;

	// Copies
	
	std::unordered_map<const Analysis::DataObject *, const Analysis::DataObject *> CopyObjects;

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
				output += parameter->GetName() + " = " + Analysis::ShapeUtils::ShapeString(shape);
			}
		}
		else
		{
			output += "-";
		}
		output += "\nParameter objects: ";
		if (ParameterObjects.size() > 0)
		{
			bool first = true;
			for (const auto& [parameter, object] : ParameterObjects)
			{
				if (!first)
				{
					output += ", ";
				}
				first = false;
				output += parameter->GetName() + " = " + object->ToString();
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
		output += "\nReturn write shapes: ";
		if (ReturnShapes.size() > 0)
		{
			bool first = true;
			for (const auto& shape : ReturnWriteShapes)
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
