#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

namespace Backend {
namespace Codegen {

class OrGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "OrGenerator"; }

	// Instruction

	void Generate(const PTX::_OrInstruction *instruction);

	template<class T>
	void Visit(const PTX::OrInstruction<T> *instruction);
};

}
}
