#pragma once

#include "Backend/Codegen/Generators/Instructions/Logical/LogicGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class XorGenerator : public LogicGenerator
{
public:
	using LogicGenerator::LogicGenerator;

	std::string Name() const override { return "XorGenerator"; }

	// Instruction

	void Generate(const PTX::_XorInstruction *instruction);

	template<class T>
	void Visit(const PTX::XorInstruction<T> *instruction);

	template<class T>
	void GenerateMaxwell(const PTX::XorInstruction<T> *instruction);

	template<class T>
	void GenerateVolta(const PTX::XorInstruction<T> *instruction);
};

}
}
