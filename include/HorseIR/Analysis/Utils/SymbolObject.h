#pragma once

#include <sstream>

#include "HorseIR/Semantics/SymbolTable/SymbolTable.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace HorseIR {
namespace Analysis {

struct SymbolObject : FlowAnalysisPointerValue<SymbolTable::Symbol>
{
	using Type = SymbolTable::Symbol;
	using FlowAnalysisPointerValue<Type>::Equals;

	struct Hash
	{
		std::size_t operator()(const Type *val) const
		{
			return std::hash<const Type *>()(val);
		}
	};

	static void Print(std::ostream& os, const Type *val)
	{
		os << PrettyPrinter::PrettyString(val->node);
	}
};

}
}
