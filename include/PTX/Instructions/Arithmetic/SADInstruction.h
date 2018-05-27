#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class SADInstruction : public InstructionBase_2<T>
{
	REQUIRE_BASE_TYPE(SADInstruction, ScalarType);
	DISABLE_EXACT_TYPE(SADInstruction, Int8Type);
	DISABLE_EXACT_TYPE(SADInstruction, UInt8Type);
	DISABLE_EXACT_TYPE_TEMPLATE(SADInstruction, BitType);
	DISABLE_EXACT_TYPE_TEMPLATE(SADInstruction, FloatType);
public:
	using InstructionBase_2<T>::InstructionBase_2;

	std::string OpCode() const
	{
		return "sad" + T::Name();
	}
};

}
