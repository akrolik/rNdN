#pragma once

#include <string>

#include "HorseIR/Tree/Tree.h"

namespace Codegen {

class NameUtils
{
public:
	constexpr static const char *GeometryThreadIndex = "$geometry$tidx";
	constexpr static const char *GeometryCellThreads = "$geometry$threads";
	constexpr static const char *GeometryListSize = "$geometry$list";

	constexpr static const char *GeometryDataSize = "$geometry$size";


	static std::string VariableName(const HorseIR::Identifier *identifier, const std::string& index = "")
	{
		if (index == "")
		{
			return identifier->GetName();
		}
		return identifier->GetName() + "$" + index;
	}

	static std::string VariableName(const HorseIR::VariableDeclaration *declaration)
	{
		return declaration->GetName();
	}

	static std::string DataAddressName(const std::string& name)
	{
		return "$data$" + name;
	}

	static std::string ReturnName(unsigned int returnIndex)
	{
		return "$return$" + std::to_string(returnIndex);
	}

	static std::string SizeName(const std::string& name)
	{
		return "$size$" + name;
	}
};

}
