#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class UnpackGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "UnpackGenerator"; }

	// Instruction

	void Generate(const PTX::_UnpackInstruction *instruction);

	template<class T, PTX::VectorSize V>
	void Visit(const PTX::UnpackInstruction<T, V> *instruction);

	template<class MOVInstruction, class T, PTX::VectorSize V>
	void GenerateInstruction(const PTX::UnpackInstruction<T, V> *instruction);
};

}
}
