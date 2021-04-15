#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class AddGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "AddGenerator"; }

	// Instruction

	void Generate(const PTX::_AddInstruction *instruction);

	template<class T>
	void Visit(const PTX::AddInstruction<T> *instruction);
};

}
}
