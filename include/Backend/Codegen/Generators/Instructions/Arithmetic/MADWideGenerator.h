#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class MADWideGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "MADWideGenerator"; }

	// Instruction

	void Generate(const PTX::_MADWideInstruction *instruction);

	template<class T>
	void Visit(const PTX::MADWideInstruction<T> *instruction);

	template<class T>
	void GenerateMaxwell(const PTX::MADWideInstruction<T> *instruction);

	template<class T>
	void GenerateVolta(const PTX::MADWideInstruction<T> *instruction);
};

}
}
