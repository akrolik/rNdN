#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

class CosineInstruction : public InstructionBase_1<Float32Type>, public FlushSubnormalModifier<Float32Type>
{
public:
	using InstructionBase_1<Float32Type>::InstructionBase_1;

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "cos.approx.ftz" + Float32Type::Name();
		}
		return "cos.approx" + Float32Type::Name();
	}
};

}
