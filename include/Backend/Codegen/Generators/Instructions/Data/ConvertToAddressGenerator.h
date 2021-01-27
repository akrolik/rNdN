#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

namespace Backend {
namespace Codegen {

class ConvertToAddressGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "ConvertToAddressGenerator"; }

	// Instruction

	void Generate(const PTX::_ConvertToAddressInstruction *instruction);

	template<PTX::Bits B, class T, class S>
	void Visit(const PTX::ConvertToAddressInstruction<B, T, S> *instruction);
};

}
}
