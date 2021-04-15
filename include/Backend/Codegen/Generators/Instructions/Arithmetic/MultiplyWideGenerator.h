#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class MultiplyWideGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "MultiplyWideGenerator"; }

	// Instruction

	void Generate(const PTX::_MultiplyWideInstruction *instruction);

	template<class T>
	void Visit(const PTX::MultiplyWideInstruction<T> *instruction);
};

}
}
