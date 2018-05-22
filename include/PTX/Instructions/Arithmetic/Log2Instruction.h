#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Arithmetic/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

class Log2Instruction : public InstructionBase<Float32Type, 1>, public FlushSubnormalModifier
{
public:
	using InstructionBase<Float32Type, 1>::InstructionBase;

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "lg2.approx.ftz" + Float32Type::Name();
		}
		return "lg2.approx" + Float32Type::Name();
	}
};

}
