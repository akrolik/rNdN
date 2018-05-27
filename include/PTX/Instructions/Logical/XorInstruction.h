#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class XorInstruction : public InstructionBase_2<T>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(XorInstruction, BitType);
	DISABLE_EXACT_TYPE(XorInstruction, Bit8Type);
public:
	using InstructionBase_2<T>::InstructionBase_2;

	std::string OpCode() const
	{
		return "xor" + T::Name();
	}
};

}
