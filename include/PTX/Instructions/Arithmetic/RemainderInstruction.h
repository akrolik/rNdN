#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class RemainderInstruction : public InstructionBase<T, 2>
{
	REQUIRE_TYPE(RemainderInstruction, ScalarType);
	DISABLE_TYPE(RemainderInstruction, Int8Type);
	DISABLE_TYPE(RemainderInstruction, UInt8Type);
	DISABLE_TYPES(RemainderInstruction, FloatType);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		return "rem" + T::Name();
	}
};

}
