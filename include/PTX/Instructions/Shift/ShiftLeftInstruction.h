#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class ShiftLeftInstruction : public InstructionBase_2<T, T, UInt32Type>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(ShiftLeftInstruction, BitType);
	DISABLE_EXACT_TYPE(ShiftLeftInstruction, PredicateType);
	DISABLE_EXACT_TYPE(ShiftLeftInstruction, Bit8Type);
public:
	using InstructionBase_2<T, T, UInt32Type>::InstructionBase_2;

	std::string OpCode() const
	{
		return "shl" + T::Name();
	}
};

}
