#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class AndInstruction : public InstructionBase<T, 2>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(AndInstruction, BitType);
	DISABLE_EXACT_TYPE(AndInstruction, Bit8Type);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		return "and" + T::Name();
	}
};

}
