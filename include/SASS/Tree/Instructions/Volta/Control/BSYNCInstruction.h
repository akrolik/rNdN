#pragma once

#include "SASS/Tree/Instructions/Volta/Control/ControlInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

namespace SASS {
namespace Volta {

class BSYNCInstruction : public ControlInstruction
{
public:
	BSYNCInstruction(std::uint8_t barrier, Predicate *controlPredicate = nullptr, bool negatePredicate = false)
		: ControlInstruction(controlPredicate, negatePredicate), m_barrier(barrier) {}

	// Properties

	std::uint8_t GetBarrier() const { return m_barrier; }
	void SetBarrier(std::uint8_t barrier) { m_barrier = barrier; }

	// Formatting

	std::string OpCode() const override { return "BSYNC"; }

	std::string Operands() const override
	{
		std::string code;

		// Optional control predicate spacing
		code += ControlInstruction::Operands();
		if (code.length() > 0)
		{
			code += ", ";
		}

		// Barrier resource
		code += "B" + std::to_string(m_barrier);

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x941;
	}

	std::uint64_t ToBinary() const override
	{
		auto code = ControlInstruction::ToBinary();

		// Barrier resource
		code |= BinaryUtils::Format(m_barrier, 16, 0xf);

		return code;
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	std::uint8_t m_barrier = 0;
};

}
}
