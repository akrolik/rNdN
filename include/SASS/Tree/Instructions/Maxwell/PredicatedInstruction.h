#pragma once

#include "SASS/Tree/Instructions/Maxwell/Instruction.h"
#include "SASS/Tree/Instructions/Maxwell/BinaryUtils.h"

#include "SASS/Tree/Operands/Predicate.h"

namespace SASS {
namespace Maxwell {

class PredicatedInstruction : public Instruction
{
public:
	using Instruction::Instruction;

	// Predicate

	const Predicate *GetPredicate() const { return m_predicate; }
	Predicate *GetPredicate() { return m_predicate; }
	bool GetNegatePredicate() const { return m_negatePredicate; }

	void SetPredicate(Predicate *predicate, bool negate = false)
	{
		m_predicate = predicate;
		m_negatePredicate = negate;
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
	Predicate *m_predicate = nullptr;
	bool m_negatePredicate = false;
};

}
}
