#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

namespace Backend {
namespace Codegen {

class ShiftRightGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "ShiftRightGenerator"; }

	// Instruction

	void Generate(const PTX::_ShiftRightInstruction *instruction);

	template<class T>
	void Visit(const PTX::ShiftRightInstruction<T> *instruction);
};

}
}
