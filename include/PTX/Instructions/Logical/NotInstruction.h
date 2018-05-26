#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class NotInstruction : public InstructionBase<T, 2>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(NotInstruction, BitType);
	DISABLE_EXACT_TYPE(NotInstruction, Bit8Type);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		return "not" + T::Name();
	}
};

}
