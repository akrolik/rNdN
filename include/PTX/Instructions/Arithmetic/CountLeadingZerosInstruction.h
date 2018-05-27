#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class CountLeadingZerosInstruction : public InstructionBase_1<UInt32Type, T>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(CountLeadingZerosInstruction, BitType);
	DISABLE_EXACT_TYPE(CountLeadingZerosInstruction, PredicateType);
	DISABLE_EXACT_TYPE(CountLeadingZerosInstruction, Bit8Type);
	DISABLE_EXACT_TYPE(CountLeadingZerosInstruction, Bit16Type);
public:
	using InstructionBase_1<UInt32Type, T>::InstructionBase;

	std::string OpCode() const
	{
		return "clz" + T::Name();
	}
};

}
