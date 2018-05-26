#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class OrInstruction : public InstructionBase<T, 2>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(OrInstruction, BitType);
	DISABLE_EXACT_TYPE(OrInstruction, Bit8Type);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		return "or" + T::Name();
	}
};

}
