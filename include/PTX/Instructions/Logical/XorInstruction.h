#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class XorInstruction : public InstructionBase<T, 2>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(XorInstruction, BitType);
	DISABLE_EXACT_TYPE(XorInstruction, Bit8Type);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		return "xor" + T::Name();
	}
};

}
