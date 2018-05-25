#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

class Exp2Instruction : public InstructionBase<Float32Type, 1>, public FlushSubnormalModifier<Float32Type>
{
public:
	using InstructionBase<Float32Type, 1>::InstructionBase;

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "ex2.approx.ftz" + Float32Type::Name();
		}
		return "ex2.approx" + Float32Type::Name();
	}
};

}
