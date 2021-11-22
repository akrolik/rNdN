#pragma once

#include "SASS/Tree/Instructions/Volta/Control/ControlInstruction.h"

namespace SASS {
namespace Volta {

class EXITInstruction : public ControlInstruction
{
public:
	using ControlInstruction::ControlInstruction;

	// Formatting

	std::string OpCode() const override { return "EXIT"; }

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x94d;
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }
};

}
}
