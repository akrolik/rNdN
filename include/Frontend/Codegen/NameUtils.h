#pragma once

#include <string>

#include "HorseIR/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

class NameUtils
{
public:
	constexpr static const char *ThreadGeometryListThreads = "$geometry$list$threads";
	constexpr static const char *ThreadGeometryListSize = "$geometry$list$size";
	constexpr static const char *ThreadGeometryListDataIndex = "$geometry$list$tidx";

	constexpr static const char *ThreadGeometryDataSize = "$geometry$size";

	constexpr static const char *SortStage = "$sort$stage";
	constexpr static const char *SortSubstage = "$sort$substage";
	constexpr static const char *SortNumStages = "$sort$num_stages";

	constexpr static const char *HashtableSize = "$hash$size";

	static std::string VariableName(const HorseIR::Identifier *identifier, bool isCell, unsigned int cellIndex, const std::string& index = "")
	{
		if (isCell)
		{
			return NameUtils::VariableName(identifier, cellIndex, index);
		}
		return NameUtils::VariableName(identifier, index);
	}

	static std::string VariableName(const HorseIR::Identifier *identifier, unsigned int cellIndex, const std::string& index = "")
	{
		if (index == "")
		{
			return identifier->GetName() + "$" + std::to_string(cellIndex);
		}
		return identifier->GetName() + "$" + std::to_string(cellIndex) + "$" + index;
	}

	static std::string VariableName(const HorseIR::Identifier *identifier, const std::string& index = "")
	{
		if (index == "")
		{
			return identifier->GetName();
		}
		return identifier->GetName() + "$" + index;
	}

	static std::string VariableName(const HorseIR::VariableDeclaration *declaration, unsigned int cellIndex, const std::string& index = "")
	{
		if (index == "")
		{
			return declaration->GetName() + "$" + std::to_string(cellIndex);
		}
		return declaration->GetName() + "$" + std::to_string(cellIndex) + "$" + index;
	}

	static std::string VariableName(const HorseIR::VariableDeclaration *declaration, const std::string& index = "")
	{
		if (index == "")
		{
			return declaration->GetName();
		}
		return declaration->GetName() + "$" + index;
	}

	static std::string ReturnName(unsigned int returnIndex)
	{
		return "$return$" + std::to_string(returnIndex);
	}

	template<class T>
	static std::string DataAddressName(const PTX::ParameterVariable<T> *parameter)
	{
		return "$data$" + parameter->GetName();
	}

	template<PTX::Bits B, class T>
	static std::string DataCellAddressName(const PTX::ParameterVariable<PTX::PointerType<B, T>> *parameter)
	{
		return "$data$" + parameter->GetName() + "$c";
	}

	template<PTX::Bits B, class T>
	static std::string DataCellAddressName(const PTX::ParameterVariable<PTX::PointerType<B, T>> *parameter, unsigned int cellIndex)
	{
		return "$data$" + parameter->GetName() + "$" + std::to_string(cellIndex);
	}

	template<class T>
	static std::string SizeName(const PTX::ParameterVariable<T> *parameter)
	{
		return "$size$" + parameter->GetName();
	}

	template<class T>
	static std::string SizeName(const PTX::ParameterVariable<T> *parameter, unsigned int cellIndex)
	{
		return "$size$" + parameter->GetName() + "$" + std::to_string(cellIndex);
	}
};

}
}
