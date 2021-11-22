#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class ShiftLeftGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "ShiftLeftGenerator"; }

	// Instruction

	void Generate(const PTX::_ShiftLeftInstruction *instruction);

	template<class T>
	void Visit(const PTX::ShiftLeftInstruction<T> *instruction);

	template<class T>
	void GenerateMaxwell(const PTX::ShiftLeftInstruction<T> *instruction);

	template<class T>
	void GenerateVolta(const PTX::ShiftLeftInstruction<T> *instruction);
};

}
}
