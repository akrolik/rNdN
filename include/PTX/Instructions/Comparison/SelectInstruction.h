#pragma once

#include <sstream>

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class SelectInstruction : public InstructionBase_3<T, T, T, PredicateType>
{
	REQUIRE_BASE_TYPE(SelectInstruction, ScalarType);
	DISABLE_EXACT_TYPE(SelectInstruction, PredicateType);
	DISABLE_EXACT_TYPE(SelectInstruction, Bit8Type);
	DISABLE_EXACT_TYPE(SelectInstruction, Int8Type);
	DISABLE_EXACT_TYPE(SelectInstruction, UInt8Type);
	DISABLE_EXACT_TYPE(SelectInstruction, Float16Type);
	DISABLE_EXACT_TYPE(SelectInstruction, Float16x2Type);
public:
	using InstructionBase_3<T, T, T, PredicateType>::InstructionBase_3;

	std::string OpCode() const
	{
		return "selp" + T::Name();
	}
};

}
