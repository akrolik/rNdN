#pragma once

#include "SASS/Tree/Instructions/Volta/Instruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Predicate.h"

namespace SASS {
namespace Volta {

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
		auto code = Instruction::ToBinary();

		// Add predicate to instruction, negate if necessary
		code |= BinaryUtils::InstructionPredicate(m_predicate, m_negatePredicate);

		return code;
	}

private:
	Predicate *m_predicate = nullptr;
	bool m_negatePredicate = false;
};

}
}
