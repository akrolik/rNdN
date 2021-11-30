#pragma once

#include "Backend/Codegen/Generators/Instructions/PredicatedInstructionGenerator.h"

#include "PTX/Tree/Tree.h"
#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Codegen {

class ConvertGenerator : public PredicatedInstructionGenerator
{
public:
	using PredicatedInstructionGenerator::PredicatedInstructionGenerator;

	std::string Name() const override { return "ConvertGenerator"; }

	// Instruction

	void Generate(const PTX::_ConvertInstruction *instruction);

	template<class D, class S>
	void Visit(const PTX::ConvertInstruction<D, S> *instruction);

	template<class D, class S>
	void GenerateMaxwell(const PTX::ConvertInstruction<D, S> *instruction);

	template<class D, class S>
	void GenerateVolta(const PTX::ConvertInstruction<D, S> *instruction);

private:
	template<class E, class T>
	E GetConversionType();

	template<class MOVInstruction, class I2IInstruction, class LOPInstruction, class PRMTInstruction, class SHRInstruction, class D, class S>
	void GenerateI2I(const PTX::ConvertInstruction<D, S> *instruction);

	template<class F2IInstruction, class D, class S>
	void GenerateF2I(const PTX::ConvertInstruction<D, S> *instruction);

	template<class F2IInstruction, class D, class S>
	void GenerateI2F(const PTX::ConvertInstruction<D, S> *instruction);

	template<class MOVInstruction, class F2IInstruction, class D, class S>
	void GenerateF2F(const PTX::ConvertInstruction<D, S> *instruction);
};

}
}
