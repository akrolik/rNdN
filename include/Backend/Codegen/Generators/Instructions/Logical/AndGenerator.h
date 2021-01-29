#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

namespace Backend {
namespace Codegen {

class AndGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "AndGenerator"; }

	// Instruction

	void Generate(const PTX::_AndInstruction *instruction);

	template<class T>
	void Visit(const PTX::AndInstruction<T> *instruction);
};

}
}
