#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class RemainderGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "RemainderGenerator"; }

	// Instruction

	void Generate(const PTX::_RemainderInstruction *instruction);

	template<class T>
	void Visit(const PTX::RemainderInstruction<T> *instruction);

	template<class T>
	void GenerateMaxwell(const PTX::RemainderInstruction<T> *instruction);

	template<class T>
	void GenerateVolta(const PTX::RemainderInstruction<T> *instruction);
};

}
}
