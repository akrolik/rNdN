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

	// Line number

	unsigned int GetLineNumber() const { return m_lineNumber; }
	void SetLineNumber(unsigned int lineNumber) { m_lineNumber = lineNumber; }

	// Operands

	void BuildCachedOperands()
	{
		m_sourceCache = std::move(GetSourceOperands());
		m_destinationCache = std::move(GetDestinationOperands());
	}

	std::vector<Operand *>& GetCachedSourceOperands() const { return m_sourceCache; }
	std::vector<Operand *>& GetCachedDestinationOperands() const { return m_destinationCache; }

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

	virtual std::uint64_t ToBinary() const = 0;
	virtual std::uint64_t ToBinaryHi() const = 0;

protected:
	unsigned int m_lineNumber = 0;
	Schedule m_schedule;

	mutable std::vector<Operand *> m_destinationCache;
	mutable std::vector<Operand *> m_sourceCache;

};

}
