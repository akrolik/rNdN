#pragma once

#include "Backend/Codegen/Generators/Instructions/Logical/LogicGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class AndGenerator : public LogicGenerator
{
public:
	using LogicGenerator::LogicGenerator;

	std::string Name() const override { return "AndGenerator"; }

	// Instruction

	void Generate(const PTX::_AndInstruction *instruction);

	template<class T>
	void Visit(const PTX::AndInstruction<T> *instruction);

	template<class T>
	void GenerateMaxwell(const PTX::AndInstruction<T> *instruction);

	template<class T>
	void GenerateVolta(const PTX::AndInstruction<T> *instruction);
};

}
}
