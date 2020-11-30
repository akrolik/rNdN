#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

class ActiveMaskInstruction : public InstructionBase_0<Bit32Type>
{
public:
	using InstructionBase_0<Bit32Type>::InstructionBase_0;

	static std::string Mnemonic() { return "activemask"; }

	std::string OpCode() const override
	{
		return Mnemonic() + Bit32Type::Name();
	}
};

}
