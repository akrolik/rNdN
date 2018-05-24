#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class CountLeadingZerosInstruction : public InstructionBase<T, 1, UInt32Type>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(CountLeadingZerosInstruction, BitType);
	DISABLE_EXACT_TYPE(CountLeadingZerosInstruction, Bit8Type);
	DISABLE_EXACT_TYPE(CountLeadingZerosInstruction, Bit16Type);
public:
	using InstructionBase<T, 1, UInt32Type>::InstructionBase;

	std::string OpCode() const
	{
		return "clz" + T::Name();
	}
};

}
