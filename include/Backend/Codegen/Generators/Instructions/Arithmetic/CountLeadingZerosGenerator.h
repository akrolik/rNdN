#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class CountLeadingZerosGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "CountLeadingZerosGenerator"; }

	// Instruction

	void Generate(const PTX::_CountLeadingZerosInstruction *instruction);

	template<class T>
	void Visit(const PTX::CountLeadingZerosInstruction<T> *instruction);

	template<class T>
	void GenerateMaxwell(const PTX::CountLeadingZerosInstruction<T> *instruction);

	template<class T>
	void GenerateVolta(const PTX::CountLeadingZerosInstruction<T> *instruction);
};

}
}
