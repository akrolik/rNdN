#pragma once

#include "Backend/Codegen/Generators/Generator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

namespace Backend {
namespace Codegen {

class PredicatedInstructionGenerator : public Generator
{
public:
	using Generator::Generator;

	// Generators

	void SetPredicatedInstruction(const PTX::PredicatedInstruction *instruction);
	void AddInstruction(SASS::PredicatedInstruction *instruction);

private:
	SASS::Predicate *m_predicate = nullptr;
	bool m_negatePredicate = false;
};

}
}
