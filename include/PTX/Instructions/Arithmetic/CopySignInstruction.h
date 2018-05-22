#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class CopySignInstruction : public InstructionBase<T, 2>
{
	REQUIRE_TYPES(CopySignInstruction, FloatType);
	DISABLE_TYPE(CopySignInstruction, Float16Type);
	DISABLE_TYPE(CopySignInstruction, Float16x2Type);
public:
	using InstructionBase<T, 2>::InstructionBase;
	
	std::string OpCode() const
	{
		return "copysign" + T::Name();
	}
};

}
