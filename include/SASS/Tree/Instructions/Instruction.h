#pragma once

#include <vector>

#include "SASS/Tree/Node.h"
#include "SASS/Tree/Operands/Operand.h"
#include "SASS/Tree/Instructions/Schedule.h"

namespace SASS {

class Instruction : public Node
{
public:
	// Schedule

	const Schedule& GetSchedule() const { return m_schedule; }
	Schedule& GetSchedule() { return m_schedule; }

	void SetSchedule(const Schedule& schedule) { m_schedule = schedule; }

	// Operands

	virtual std::vector<Operand *> GetSourceOperands() const = 0;
	virtual std::vector<Operand *> GetDestinationOperands() const = 0;

	// Hardware instruction properties

	enum InstructionClass {
		S2R,
		Control,
		Integer,
		SinglePrecision,
		DoublePrecision,
		SpecialFunction,
		Comparison,
		Shift,
		SharedMemoryLoad,
		SharedMemoryStore,
		GlobalMemoryLoad,
		GlobalMemoryStore,
		SCHI
	};

	virtual InstructionClass GetInstructionClass() const = 0;

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

protected:
	Schedule m_schedule;
};

}
