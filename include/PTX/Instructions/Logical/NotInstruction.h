#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class NotInstruction : public InstructionBase_2<T>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(NotInstruction, BitType);
	DISABLE_EXACT_TYPE(NotInstruction, Bit8Type);
public:
	using InstructionBase_2<T>::InstructionBase_2;

	std::string OpCode() const override
	{
		return "not" + T::Name();
	}
};

}
