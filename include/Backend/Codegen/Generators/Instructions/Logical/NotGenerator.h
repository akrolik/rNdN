#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class NotGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "NotGenerator"; }

	// Instruction

	void Generate(const PTX::_NotInstruction *instruction);

	template<class T>
	void Visit(const PTX::NotInstruction<T> *instruction);

	template<class T>
	void GenerateMaxwell(const PTX::NotInstruction<T> *instruction);

	template<class T>
	void GenerateVolta(const PTX::NotInstruction<T> *instruction);
};

}
}
