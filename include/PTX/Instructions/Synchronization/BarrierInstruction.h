#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Type.h"
#include "PTX/Operands/Operand.h"

namespace PTX {

class BarrierInstruction : public PredicatedInstruction
{
public:
	BarrierInstruction(const TypedOperand<UInt32Type> *barrier, bool aligned = false) : m_barrier(barrier), m_wait(true), m_aligned(aligned) {}
	BarrierInstruction(const TypedOperand<UInt32Type> *barrier, const TypedOperand<UInt32Type> *threads, bool wait, bool aligned = false) : m_barrier(barrier), m_threads(threads), m_wait(wait), m_aligned(aligned) {}

	const TypedOperand<UInt32Type> *GetBarrier() const { return m_barrier; }
	void SetBarrier(const TypedOperand<UInt32Type> *barrier) { m_barrier = barrier; }

	const TypedOperand<UInt32Type> *GetThreads() const { return m_threads; }
	void SetThreads(const TypedOperand<UInt32Type> *threads) { m_threads = threads; }

	bool GetWait() { return m_wait; }
	void SetWait(bool wait) { m_wait = wait; }

	bool GetAligned() { return m_aligned; }
	void SetAligned(bool aligned) { m_aligned = aligned; }

	static std::string Mnemonic() { return "barrier"; }

	std::string OpCode() const override
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

	std::vector<const Operand *> Operands() const override
	{
		if (m_wait == false || m_threads != nullptr)
		{
			return { m_barrier, m_threads };

		}
		return { m_barrier };
	}

private:
	const TypedOperand<UInt32Type> *m_barrier = nullptr;
	const TypedOperand<UInt32Type> *m_threads = nullptr;
	bool m_wait = false;
	bool m_aligned = false;
};

}
