#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

class Log2Instruction : public InstructionBase_1<Float32Type>, public FlushSubnormalModifier<Float32Type>
{
public:
	using InstructionBase_1<Float32Type>::InstructionBase_1;

	std::string OpCode() const override
	{
		return "lg2.approx" + FlushSubnormalModifier<Float32Type>::OpCodeModifier() + Float32Type::Name();
	}
};

}
