#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class JoinCountGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "JoinCountGenerator"; }

	void Generate(const std::vector<HorseIR::LValue *>& targets, const std::vector<HorseIR::Operand *>& arguments)
	{
		//TODO: @join_count
	}
};

}
