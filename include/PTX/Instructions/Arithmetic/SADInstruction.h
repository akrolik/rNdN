#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class SADInstruction : public InstructionBase<T, 2>
{
	REQUIRE_TYPE(SADInstruction, ScalarType);
	DISABLE_TYPE(SADInstruction, Int8Type);
	DISABLE_TYPE(SADInstruction, UInt8Type);
	DISABLE_TYPES(SADInstruction, FloatType);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		return "sad" + T::Name();
	}
};

}
