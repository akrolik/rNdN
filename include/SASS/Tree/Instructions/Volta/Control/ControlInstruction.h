#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

namespace SASS {
namespace Volta {

class ControlInstruction : public PredicatedInstruction
{
public:
	ControlInstruction(Predicate *controlPredicate = nullptr, bool negatePredicate = false)
		: m_controlPredicate(controlPredicate), m_negateControlPredicate(negatePredicate) {}

	// Properties

	const Predicate *GetControlPredicate() const { return m_controlPredicate; }
	Predicate *GetControlPredicate() { return m_controlPredicate; }
	bool GetNegateControlPredicate() const { return m_negateControlPredicate; }

	void SetControlPredicate(Predicate *controlPredicate, bool negate = false)
	{
		m_controlPredicate = controlPredicate;
		m_negateControlPredicate = negate;
	}

	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { m_controlPredicate };
	}

	// Formatting

	std::string Operands() const override
	{
		std::string code;
		if (m_controlPredicate != nullptr)
		{
			if (m_negateControlPredicate)
			{
				code += "!";
			}
			code += m_controlPredicate->ToString();
		}
		return code;
	}

	// Binary

	std::uint64_t ToBinaryHi() const override
	{
		auto code = PredicatedInstruction::ToBinaryHi();

		// Control predicate

		code |= BinaryUtils::ControlPredicate(m_controlPredicate, m_negateControlPredicate);

		return code;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::Control; }

protected:
	Predicate *m_controlPredicate = nullptr;
	bool m_negateControlPredicate = false;
};

}
}
