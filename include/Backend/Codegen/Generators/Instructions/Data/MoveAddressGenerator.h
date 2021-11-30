#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class MoveAddressGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "MoveAddressGenerator"; }

	// Instruction

	void Generate(const PTX::_MoveAddressInstruction *instruction);

	template<PTX::Bits B, class T, class S>
	void Visit(const PTX::MoveAddressInstruction<B, T, S> *instruction);

	template<class MOVInstruction, PTX::Bits B, class T, class S>
	void GenerateInstruction(const PTX::MoveAddressInstruction<B, T, S> *instruction);
};

}
}
