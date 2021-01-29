#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

namespace Backend {
namespace Codegen {

class XorGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "XorGenerator"; }

	// Instruction

	void Generate(const PTX::_XorInstruction *instruction);

	template<class T>
	void Visit(const PTX::XorInstruction<T> *instruction);
};

}
}
