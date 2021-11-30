#pragma once

#include "SASS/Tree/Instructions/Volta/Control/DivergenceInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

namespace SASS {
namespace Volta {

class BSSYInstruction : public DivergenceInstruction
{
public:
	BSSYInstruction(const std::string& target, std::uint8_t barrier, Predicate *controlPredicate = nullptr, bool negatePredicate = false)
		: DivergenceInstruction(target, controlPredicate, negatePredicate), m_barrier(barrier) {}

	// Properties

	std::uint8_t GetBarrier() const { return m_barrier; }
	void SetBarrier(std::uint8_t barrier) { m_barrier = barrier; }

	// Formatting

	std::string OpCode() const override { return "BSSY"; }

	std::string Operands() const override
	{
		std::string code;

		// Optional control predicate spacing
		code += ControlInstruction::Operands();
		if (code.length() > 0)
		{
			code += ", ";
		}

		// Barrier resource, between control predicate and address
		code += "B" + std::to_string(m_barrier);
		code += ", ";

		// Absolute target address
		code += "`(" + this->GetTarget() + ") [" + Utils::Format::HexString(this->GetTargetAddress(), 4) + "]";

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x945;
	}

	std::uint64_t ToBinary() const override
	{
		auto code = DivergenceInstruction::ToBinary();

		// Barrier resource
		code |= BinaryUtils::Format(m_barrier, 16, 0xf);

		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		// Do not include Hi bits of address

		return ControlInstruction::ToBinaryHi();
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	std::uint8_t m_barrier = 0;
};

}
}
