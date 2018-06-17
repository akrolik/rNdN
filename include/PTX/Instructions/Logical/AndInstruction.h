#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class AndInstruction : public InstructionBase_2<T>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(AndInstruction, BitType);
	DISABLE_EXACT_TYPE(AndInstruction, Bit8Type);
public:
	using InstructionBase_2<T>::InstructionBase_2;

	std::string OpCode() const override
	{
		return "and" + T::Name();
	}
};

}
