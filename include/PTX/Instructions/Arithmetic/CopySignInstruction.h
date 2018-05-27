#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class CopySignInstruction : public InstructionBase_2<T>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(CopySignInstruction, FloatType);
	DISABLE_EXACT_TYPE(CopySignInstruction, Float16Type);
	DISABLE_EXACT_TYPE(CopySignInstruction, Float16x2Type);
public:
	using InstructionBase_2<T>::InstructionBase;
	
	std::string OpCode() const
	{
		return "copysign" + T::Name();
	}
};

}
