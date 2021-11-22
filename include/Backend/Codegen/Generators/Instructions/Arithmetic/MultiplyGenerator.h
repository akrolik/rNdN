#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class MultiplyGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "MultiplyGenerator"; }

	// Instruction

	void Generate(const PTX::_MultiplyInstruction *instruction);

	template<class T>
	void Visit(const PTX::MultiplyInstruction<T> *instruction);

	template<class T>
	void GenerateMaxwell(const PTX::MultiplyInstruction<T> *instruction);

	template<class T>
	void GenerateVolta(const PTX::MultiplyInstruction<T> *instruction);
};

}
}
