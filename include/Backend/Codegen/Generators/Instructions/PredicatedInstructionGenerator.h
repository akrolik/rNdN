#pragma once

#include "Backend/Codegen/Generators/Generator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class PredicatedInstructionGenerator : public Generator
{
public:
	using Generator::Generator;

	// Generators

	void SetPredicatedInstruction(const PTX::PredicatedInstruction *instruction);

	void AddInstruction(SASS::Maxwell::PredicatedInstruction *instruction, SASS::Predicate *predicate = nullptr, bool negatePredicate = false);
	void AddInstruction(SASS::Volta::PredicatedInstruction *instruction, SASS::Predicate *predicate = nullptr, bool negatePredicate = false);

private:
	SASS::Predicate *m_predicate = nullptr;
	bool m_negatePredicate = false;

	const PTX::PredicatedInstruction *m_instruction = nullptr;

	template<class I>
	void _AddInstruction(I *instruction, SASS::Predicate *predicate, bool negatePredicate);
};

}
}
