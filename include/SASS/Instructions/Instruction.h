#pragma once

#include <vector>

#include "SASS/Node.h"
#include "SASS/Operands/Operand.h"

#include "SASS/Traversal/Visitor.h"

namespace SASS {

class Instruction : public Node
{
public:
	class SCHI {
	public:
		SCHI() {}
		SCHI(std::uint8_t stall, bool yield, std::uint8_t writeBarrier, std::uint8_t readBarrier, std::uint8_t waitBarriers, std::uint8_t reuse)
			: m_stall(stall), m_yield(yield), m_writeBarrier(writeBarrier), m_readBarrier(readBarrier), m_waitBarriers(waitBarriers), m_reuse(reuse) {}

		void SetScheduling(std::uint8_t stall, bool yield, std::uint8_t writeBarrier, std::uint8_t readBarrier, std::uint8_t waitBarriers, std::uint8_t reuse)
		{
			m_stall = stall;
			m_yield = yield;
			m_writeBarrier = writeBarrier;
			m_readBarrier = readBarrier;
			m_waitBarriers = waitBarriers;
			m_reuse = reuse;
		}

		std::string ToString() const
		{
			std::string code;
			code += "Stall=" + std::to_string(m_stall) + "; ";
			code += "Yield=" + std::to_string(m_yield) + "; ";
			code += "WriteB=" + std::to_string(m_writeBarrier) + "; ";
			code += "ReadB=" + std::to_string(m_readBarrier) + "; ";
			code += "WaitB=" + std::to_string(m_waitBarriers) + "; ";
			code += "Reuse=" + std::to_string(m_reuse);
			return code;
		}

		std::uint32_t GenCode() const
		{
			std::uint32_t code = 0u;
			code |= (m_reuse        << 17);
			code |= (m_waitBarriers << 11);
			code |= (m_readBarrier  << 8);
			code |= (m_writeBarrier << 5);
			code |= (m_yield        << 4);
			code |= (m_stall        << 0);
			return code;
		}

	private:
		std::uint8_t m_stall = 0;
		bool m_yield = false;
		std::uint8_t m_writeBarrier = 0;
		std::uint8_t m_readBarrier = 0;
		std::uint8_t m_waitBarriers = 0;
		std::uint8_t m_reuse = 0;
	};

	Instruction() {}
	Instruction(const std::vector<const Operand *>& operands) : m_operands(operands) {}

	// Scheduling

	const SCHI& GetScheduling() const { return m_scheduling; }
	SCHI& GetScheduling() { return m_scheduling; }
	void SetScheduling(std::uint8_t stall, bool yield, std::uint8_t writeBarrier, std::uint8_t readBarrier, std::uint8_t waitBarriers, std::uint8_t reuse)
	{
		m_scheduling.SetScheduling(stall, yield, writeBarrier, readBarrier, waitBarriers, reuse);
	}

	// Formatting

	virtual std::string OpCode() const = 0;
	virtual std::string OpModifiers() const { return ""; }
	virtual std::string Operands() const { return ""; }

	std::string ToString() const override
	{
		return OpCode() + OpModifiers() + " " + Operands() + ";";
	}

	// Binary

	virtual std::uint64_t BinaryOpCode() const = 0;
	virtual std::uint64_t BinaryOpModifiers() const { return 0; }
	virtual std::uint64_t BinaryOperands() const { return 0; }

	virtual std::uint64_t ToBinary() const
	{
		std::uint64_t code = 0x0;
		code |= BinaryOpCode();
		code |= BinaryOpModifiers();
		code |= BinaryOperands();
		return code;
	}

	// Visitors

	virtual void Accept(Visitor& visitor) = 0;

private:
	std::vector<const Operand *> m_operands;
	SCHI m_scheduling;
};

}
