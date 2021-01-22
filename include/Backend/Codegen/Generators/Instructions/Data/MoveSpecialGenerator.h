#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

namespace Backend {
namespace Codegen {

class MoveSpecialGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "MoveSpecialGenerator"; }

	// Generators

	void Generate(const PTX::_MoveSpecialInstruction *instruction);

	template<class T>
	void Visit(const PTX::MoveSpecialInstruction<T> *instruction);
};

}
}
