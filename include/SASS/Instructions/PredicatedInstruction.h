#pragma once

#include "SASS/Instructions/Instruction.h"
#include "SASS/Operands/Predicate.h"

namespace SASS {

class PredicatedInstruction : public Instruction
{
public:
	using Instruction::Instruction;

	// Predicate

	void SetPredicate(const Predicate *predicate, bool negate = false)
	{
		m_predicate = predicate;
		m_negatePredicate = negate;
	}

	const Predicate *GetPredicate() const { return m_predicate; }
	bool GetNegatePredicate() const { return m_negatePredicate; }

	// Formatting

	std::string ToString() const override
	{
		std::string code;
		if (m_predicate != nullptr)
		{
			code += "@";
			if (m_negatePredicate)
			{
				code += "!";
			}
			code += m_predicate->ToString() + " ";
		}
		return code + Instruction::ToString();
	}

	// Binary

	std::uint64_t ToBinary() const override
	{
		// Add predicate to instruction, negate if necessary

		auto code = Instruction::ToBinary();
		if (m_predicate != nullptr)
		{
			code |= BinaryUtils::InstructionPredicate(m_predicate);
			if (m_negatePredicate)
			{
				code |= BinaryUtils::Format(0x8, 16, 0x8);
			}
		}
		else
		{
			code |= BinaryUtils::Format(0x7, 16, 0x7);
		}
		return code;
	}

private:
	const Predicate *m_predicate = nullptr;
	bool m_negatePredicate = false;
};

}
