#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

class Exp2Instruction : public InstructionBase_1<Float32Type>, public FlushSubnormalModifier<Float32Type>
{
public:
	using InstructionBase_1<Float32Type>::InstructionBase_1;

	std::string OpCode() const override
	{
		if (m_flush)
		{
			return "ex2.approx.ftz" + Float32Type::Name();
		}
		return "ex2.approx" + Float32Type::Name();
	}
};

}
