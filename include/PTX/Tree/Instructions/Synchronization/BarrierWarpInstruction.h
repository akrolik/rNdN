#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Operands/Constants/Value.h"

namespace PTX {

class BarrierWarpInstruction : public PredicatedInstruction
{
public:
	BarrierWarpInstruction(UInt32Value *memberMask) : m_memberMask(memberMask) {}

	// Analysis properties

	bool HasSideEffect() const override { return true; }

	// Properties

	const UInt32Value *GetMemberMask() const { return m_memberMask; }
	UInt32Value *GetMemberMask() { return m_memberMask; }
	void SetMemberMask(UInt32Value *memberMask) { m_memberMask = memberMask; }

	// Formatting

	static std::string Mnemonic() { return "bar.warp"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + ".sync";
	}

	std::vector<const Operand *> GetOperands() const override
	{
		return { m_memberMask };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { m_memberMask };
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	UInt32Value *m_memberMask = nullptr;
};

}
