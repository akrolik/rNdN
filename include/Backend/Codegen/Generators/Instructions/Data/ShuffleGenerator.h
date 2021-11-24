#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class ShuffleGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "ShuffleGenerator"; }

	// Instruction

	void Generate(const PTX::_ShuffleInstruction *instruction);

	template<class T>
	void Visit(const PTX::ShuffleInstruction<T> *instruction);

	template<class SHFLInstruction, class T>
	void GenerateInstruction(const PTX::ShuffleInstruction<T> *instruction);
};

}
}
