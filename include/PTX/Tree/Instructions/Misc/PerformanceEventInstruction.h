#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Operands/Constants/Value.h"

namespace PTX {

class PerformanceEventInstruction : public PredicatedInstruction, public DispatchBase
{
public:
	PerformanceEventInstruction(UInt32Value *eventMask, bool mask = false) : m_eventMask(eventMask), m_mask(mask) {}

	// Analysis properties

	bool HasSideEffect() const override { return true; }

	// Properties

	const UInt32Value *GetEventMask() const { return m_eventMask; }
	UInt32Value *GetEventMask() { return m_eventMask; }
	void SetEventMask(UInt32Value *eventMask) { m_eventMask = eventMask; }

	bool IsMask() const { return m_mask; }
	void SetMask(bool mask) { m_mask = mask; }

	// Formatting

	static std::string Mnemonic() { return "pmevent"; }

	std::string GetOpCode() const override
	{
		if (m_mask)
		{
			return Mnemonic() + ".mask";
		}
		return Mnemonic();
	}

	std::vector<const Operand *> GetOperands() const override
	{
		return { m_eventMask };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { m_eventMask };
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	UInt32Value *m_eventMask = nullptr;
	bool m_mask = false;
};

}
