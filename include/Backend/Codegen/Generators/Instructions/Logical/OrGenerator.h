#pragma once

#include "Backend/Codegen/Generators/Instructions/Logical/LogicGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class OrGenerator : public LogicGenerator
{
public:
	using LogicGenerator::LogicGenerator;

	std::string Name() const override { return "OrGenerator"; }

	// Instruction

	void Generate(const PTX::_OrInstruction *instruction);

	template<class T>
	void Visit(const PTX::OrInstruction<T> *instruction);

	template<class T>
	void GenerateMaxwell(const PTX::OrInstruction<T> *instruction);

	template<class T>
	void GenerateVolta(const PTX::OrInstruction<T> *instruction);
};

}
}
