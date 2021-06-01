#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Type.h"
#include "PTX/Tree/Operands/Operand.h"

namespace PTX {

class BarrierInstruction : public PredicatedInstruction
{
public:
	BarrierInstruction(TypedOperand<UInt32Type> *barrier, bool aligned = false) : m_barrier(barrier), m_wait(true), m_aligned(aligned) {}
	BarrierInstruction(TypedOperand<UInt32Type> *barrier, TypedOperand<UInt32Type> *threads, bool wait, bool aligned = false) : m_barrier(barrier), m_threads(threads), m_wait(wait), m_aligned(aligned) {}

	// Analysis properties

	bool HasSideEffect() const override { return true; }

	// Properties

	const TypedOperand<UInt32Type> *GetBarrier() const { return m_barrier; }
	TypedOperand<UInt32Type> *GetBarrier() { return m_barrier; }
	void SetBarrier(TypedOperand<UInt32Type> *barrier) { m_barrier = barrier; }

	const TypedOperand<UInt32Type> *GetThreads() const { return m_threads; }
	TypedOperand<UInt32Type> *GetThreads() { return m_threads; }
	void SetThreads(TypedOperand<UInt32Type> *threads) { m_threads = threads; }

	bool GetWait() const { return m_wait; }
	void SetWait(bool wait) { m_wait = wait; }

	bool GetAligned() const { return m_aligned; }
	void SetAligned(bool aligned) { m_aligned = aligned; }

	// Formatting

	static std::string Mnemonic() { return "barrier"; }

	std::string GetOpCode() const override
	{
		std::string code = Mnemonic();
		if (m_wait)
		{
			code += ".sync";
		}
		else
		{
			code += ".arrive";
		}
		if (m_aligned)
		{
			code += ".aligned";
		}
		return code;
	}

	std::vector<const Operand *> GetOperands() const override
	{
		if (m_wait == false || m_threads != nullptr)
		{
			return { m_barrier, m_threads };
		}
		return { m_barrier };
	}

	std::vector<Operand *> GetOperands() override
	{
		if (m_wait == false || m_threads != nullptr)
		{
			return { m_barrier, m_threads };
		}
		return { m_barrier };
	}

	// Visitors
	
	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	TypedOperand<UInt32Type> *m_barrier = nullptr;
	TypedOperand<UInt32Type> *m_threads = nullptr;
	bool m_wait = false;
	bool m_aligned = false;
};

}
