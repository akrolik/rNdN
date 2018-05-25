#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

class CosineInstruction : public InstructionBase<Float32Type, 1>, public FlushSubnormalModifier<Float32Type>
{
public:
	using InstructionBase<Float32Type, 1>::InstructionBase;

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
