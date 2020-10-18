#pragma once

#include <vector>

#include "SASS/Node.h"
#include "SASS/Operands/Operand.h"

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
			code |= m_reuse;
			code <<= 6;
			code |= m_waitBarriers;
			code <<= 3;
			code |= m_readBarrier;
			code <<= 3;
			code |= m_writeBarrier;
			code <<= 1;
			code |= m_yield;
			code <<= 4;
			code |= m_stall;
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

	virtual std::string OpCode() const = 0;
	virtual std::string OpModifiers() const { return ""; }
	virtual std::string OperandModifiers(unsigned int index) const { return ""; }

	const SCHI& GetScheduling() const { return m_scheduling; }
	void SetScheduling(std::uint8_t stall, bool yield, std::uint8_t writeBarrier, std::uint8_t readBarrier, std::uint8_t waitBarriers, std::uint8_t reuse)
	{
		m_scheduling.SetScheduling(stall, yield, writeBarrier, readBarrier, waitBarriers, reuse);
	}

	std::string ToString() const override
	{
		std::string code = OpCode() + OpModifiers();

		auto index = 0u;
		for (const auto& operand : m_operands)
		{
			if (index == 0)
			{
				code += " ";
			}
			else
			{
				code += ", ";
			}
			code += operand->ToString();
			code += OperandModifiers(index++);
		}

		return code + " ;";
	}

private:
	std::vector<const Operand *> m_operands;
	SCHI m_scheduling;
};

}
