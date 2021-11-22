#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class MoveGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "MoveGenerator"; }

	// Instruction

	void Generate(const PTX::_MoveInstruction *instruction);

	template<class T>
	void Visit(const PTX::MoveInstruction<T> *instruction);

	template<class T>
	void GenerateMaxwell(const PTX::MoveInstruction<T> *instruction);

	template<class T>
	void GenerateVolta(const PTX::MoveInstruction<T> *instruction);
};

}
}
