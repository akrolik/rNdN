#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class CNotInstruction : public InstructionBase_2<T>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(CNotInstruction, BitType);
	DISABLE_EXACT_TYPE(CNotInstruction, PredicateType);
	DISABLE_EXACT_TYPE(CNotInstruction, Bit8Type);
public:
	using InstructionBase_2<T>::InstructionBase_2;

	std::string OpCode() const
	{
		return "cnot" + T::Name();
	}
};

}
