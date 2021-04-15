#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class MADGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "MADGenerator"; }

	// Instruction

	void Generate(const PTX::_MADInstruction *instruction);

	template<class T>
	void Visit(const PTX::MADInstruction<T> *instruction);
};

}
}
