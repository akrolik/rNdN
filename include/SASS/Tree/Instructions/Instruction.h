#pragma once

#include <vector>

#include "SASS/Tree/Node.h"
#include "SASS/Tree/Operands/Operand.h"

namespace SASS {

class Instruction : public Node
{
public:
	class SCHI {
	public:
		SCHI() {}
		SCHI(std::uint8_t stall, bool yield, std::uint8_t writeBarrier, std::uint8_t readBarrier, std::uint8_t waitBarriers, std::uint8_t reuse)
			: m_stall(stall), m_yield(yield), m_writeBarrier(writeBarrier), m_readBarrier(readBarrier), m_waitBarriers(waitBarriers), m_reuse(reuse) {}

		void SetSchedule(std::uint8_t stall, bool yield, std::uint8_t writeBarrier, std::uint8_t readBarrier, std::uint8_t waitBarriers, std::uint8_t reuse)
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

	// Schedule

	const SCHI& GetSchedule() const { return m_schedule; }
	SCHI& GetSchedule() { return m_schedule; }
	void SetSchedule(std::uint8_t stall, bool yield, std::uint8_t writeBarrier, std::uint8_t readBarrier, std::uint8_t waitBarriers, std::uint8_t reuse)
	{
		m_schedule.SetSchedule(stall, yield, writeBarrier, readBarrier, waitBarriers, reuse);
	}

	// Operands

	virtual std::vector<Operand *> GetSourceOperands() const = 0;
	virtual std::vector<Operand *> GetDestinationOperands() const = 0;

	// Hardware properties

	enum HardwareClass {
		S2R,
		SharedMemory,
		GlobalMemory,
		x32,
		x64,
		qtr, //TODO: Name
		Shift,
		Compare,
		Schedule
	};

	virtual HardwareClass GetHardwareClass() const = 0;

	// Formatting

	virtual std::string OpCode() const = 0;
	virtual std::string OpModifiers() const { return ""; }
	virtual std::string Operands() const { return ""; }

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

private:
	SCHI m_schedule;
};

}
