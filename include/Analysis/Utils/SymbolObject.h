#pragma once

#include <sstream>

#include "HorseIR/Semantics/SymbolTable/SymbolTable.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace Analysis {

struct SymbolObject : HorseIR::FlowAnalysisPointerValue<HorseIR::SymbolTable::Symbol>
{
	using Type = HorseIR::SymbolTable::Symbol;
	using HorseIR::FlowAnalysisPointerValue<Type>::Equals;

	struct Hash
	{
		std::size_t operator()(const Type *val) const
		{
			return std::hash<const Type *>()(val);
		}
	};

	static void Print(std::ostream& os, const Type *val)
	{
		os << HorseIR::PrettyPrinter::PrettyString(val->node);
	}
};

}
