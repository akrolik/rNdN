#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class RemainderInstruction : public InstructionBase_2<T>
{
	REQUIRE_BASE_TYPE(RemainderInstruction, ScalarType);
	DISABLE_EXACT_TYPE(RemainderInstruction, Int8Type);
	DISABLE_EXACT_TYPE(RemainderInstruction, UInt8Type);
	DISABLE_EXACT_TYPE_TEMPLATE(RemainderInstruction, BitType);
	DISABLE_EXACT_TYPE_TEMPLATE(RemainderInstruction, FloatType);
public:
	using InstructionBase_2<T>::InstructionBase_2;

	std::string OpCode() const
	{
		return "rem" + T::Name();
	}
};

}
