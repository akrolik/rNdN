#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class PackGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "PackGenerator"; }

	// Instruction

	void Generate(const PTX::_PackInstruction *instruction);

	template<class T, PTX::VectorSize V>
	void Visit(const PTX::PackInstruction<T, V> *instruction);
};

}
}
