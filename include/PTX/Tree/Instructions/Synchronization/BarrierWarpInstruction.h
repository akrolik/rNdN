#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Operands/Extended/HexOperand.h"

namespace PTX {

class BarrierWarpInstruction : public PredicatedInstruction
{
public:
	BarrierWarpInstruction(std::uint32_t memberMask) : m_memberMask(memberMask) {}

	// Properties

	std::uint32_t GetMemberMask() const { return m_memberMask; }
	void SetMemberMask(std::uint32_t memberMask) { m_memberMask = memberMask; }

	// Formatting

	static std::string Mnemonic() { return "bar.warp"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + ".sync";
	}

	std::vector<const Operand *> GetOperands() const override
	{
		return { new HexOperand(m_memberMask) };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { new HexOperand(m_memberMask) };
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	std::uint32_t m_memberMask = 0;
};

}
