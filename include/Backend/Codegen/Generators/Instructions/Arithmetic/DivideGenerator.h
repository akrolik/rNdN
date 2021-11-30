#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class DivideGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "DivideGenerator"; }

	// Instruction

	void Generate(const PTX::_DivideInstruction *instruction);

	template<class T>
	void Visit(const PTX::DivideInstruction<T> *instruction);

	template<class T>
	void GenerateMaxwell(const PTX::DivideInstruction<T> *instruction);

	template<class T>
	void GenerateVolta(const PTX::DivideInstruction<T> *instruction);

private:
	template<class MOVInstruction, class DMULInstruction, class DFMAInstruction, class MUFUInstruction>
	void GenerateF64(const PTX::DivideInstruction<PTX::Float64Type> *instruction);
};

}
}
