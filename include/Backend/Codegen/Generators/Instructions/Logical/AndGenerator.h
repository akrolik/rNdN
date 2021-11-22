#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

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

	template<class T>
	void GenerateMaxwell(const PTX::AndInstruction<T> *instruction);

	template<class T>
	void GenerateVolta(const PTX::AndInstruction<T> *instruction);
};

}
}
