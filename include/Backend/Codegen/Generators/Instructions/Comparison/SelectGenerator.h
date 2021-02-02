#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

namespace Backend {
namespace Codegen {

class SelectGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "SelectGenerator"; }

	// Instruction

	void Generate(const PTX::_SelectInstruction *instruction);

	template<class T>
	void Visit(const PTX::SelectInstruction<T> *instruction);
};

}
}