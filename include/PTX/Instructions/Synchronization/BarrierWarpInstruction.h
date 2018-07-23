#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Extended/HexOperand.h"

namespace PTX {

class BarrierWarpInstruction : public PredicatedInstruction
{
public:
	BarrierWarpInstruction(uint32_t memberMask) : m_memberMask(memberMask) {}

	uint32_t GetMemberMask() const { return m_memberMask; }
	void SetMemberMask(uint32_t memberMask) { m_memberMask = memberMask; }

	static std::string Mnemonic() { return "bar.warp"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ".sync";
	}

	std::vector<const Operand *> Operands() const override
	{
		return { new HexOperand(m_memberMask) };
	}

private:
	uint32_t m_memberMask = 0;
};

}
