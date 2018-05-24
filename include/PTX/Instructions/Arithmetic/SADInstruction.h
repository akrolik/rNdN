#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class SADInstruction : public InstructionBase<T, 2>
{
	REQUIRE_BASE_TYPE(SADInstruction, ScalarType);
	DISABLE_EXACT_TYPE(SADInstruction, Int8Type);
	DISABLE_EXACT_TYPE(SADInstruction, UInt8Type);
	DISABLE_EXACT_TYPE_TEMPLATE(SADInstruction, BitType);
	DISABLE_EXACT_TYPE_TEMPLATE(SADInstruction, FloatType);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		return "sad" + T::Name();
	}
};

}
