#pragma once

#include "Codegen/Builder.h"
#include "Codegen/ResourceAllocator.h"
#include "Codegen/Generators/AddressGenerator.h"

#include "HorseIR/Tree/Parameter.h"

#include "PTX/Type.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Functions/DataFunction.h"
#include "PTX/Instructions/Data/LoadInstruction.h"
#include "PTX/Statements/BlockStatement.h"

namespace Codegen {

template<PTX::Bits B>
class ParameterGenerator
{
public:
	using NodeType = HorseIR::Parameter;

	template<class T>
	static void Generate(HorseIR::Parameter *parameter, Builder *builder)
	{
		//TODO: Remove check for loading parameters
		if constexpr(!std::is_same<T, PTX::PredicateType>::value)
		{
		auto declaration = new PTX::PointerDeclaration<B, T>(parameter->GetName());
		builder->AddParameter(declaration);
		auto variable = declaration->GetVariable(parameter->GetName());
		auto value = builder->AllocateRegister<T>(parameter->GetName());

		auto block = new PTX::BlockStatement();
		builder->AddStatement(block);
		builder->OpenScope(block);

		auto address = AddressGenerator<B>::template Generate<T>(variable, builder);
		builder->AddStatement(new PTX::LoadInstruction<B, T, PTX::GlobalSpace>(value, address));

		builder->CloseScope();
		}
	}
};

}
