#pragma once

#include <string>

namespace Codegen {

class NameUtils
{
public:
	constexpr static const char *GeometryThreadIndex = "$geometry$tidx";
	constexpr static const char *GeometryCellThreads = "$geometry$threads";
	constexpr static const char *GeometryListSize = "$geometry$list";
	constexpr static const char *GeometryCellSize = "$geometry$cell";

	constexpr static const char *GeometryVectorSize = "$geometry$size";

	static std::string VariableName(const std::string& name, const std::string& index = "")
	{
		if (index == "")
		{
			return name;
		}
		return name + "$" + index;
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
