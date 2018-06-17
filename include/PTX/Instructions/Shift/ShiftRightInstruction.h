#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class ShiftRightInstruction : public InstructionBase_2<T, T, UInt32Type>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(ShiftRightInstruction, BitType);
	DISABLE_EXACT_TYPE(ShiftRightInstruction, PredicateType);
	DISABLE_EXACT_TYPE(ShiftRightInstruction, Bit8Type);
public:
	using InstructionBase_2<T, T, UInt32Type>::InstructionBase_2;

	std::string OpCode() const override
	{
		return "shr" + T::Name();
	}
};

}
