#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

class SineInstruction : public InstructionBase_1<Float32Type>, public FlushSubnormalModifier<Float32Type>
{
public:
	using InstructionBase_1<Float32Type>::InstructionBase_1;

	std::string OpCode() const override
	{
		if (m_flush)
		{
			return "sin.approx.ftz" + Float32Type::Name();
		}
		return "sin.approx" + Float32Type::Name();
	}
};

}
