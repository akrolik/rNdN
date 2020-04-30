#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Data/TargetCellGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class JoinGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "JoinGenerator"; }

	void Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		TargetCellGenerator<B, PTX::Int64Type> targetGenerator(this->m_builder);
		auto targetRegister0 = targetGenerator.Generate(target, 0);
		auto targetRegister1 = targetGenerator.Generate(target, 1);
		//TODO: @join
	}
};

}
