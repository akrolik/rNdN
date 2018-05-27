#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class OrInstruction : public InstructionBase_2<T>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(OrInstruction, BitType);
	DISABLE_EXACT_TYPE(OrInstruction, Bit8Type);
public:
	using InstructionBase_2<T>::InstructionBase_2;

	std::string OpCode() const
	{
		return "or" + T::Name();
	}
};

}
