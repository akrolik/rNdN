#pragma once

#include "Codegen/Builder.h"
#include "Codegen/ResourceAllocator.h"
#include "Codegen/Generators/AddressGenerator.h"

#include "HorseIR/Tree/Parameter.h"

#include "PTX/Type.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Functions/DataFunction.h"
#include "PTX/Instructions/Comparison/SetPredicateInstruction.h"
#include "PTX/Instructions/Data/ConvertInstruction.h"
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
		//TODO: Specialize?
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			auto declaration = new PTX::PointerDeclaration<B, PTX::Int8Type>(parameter->GetName());
			builder->AddParameter(declaration);
			auto variable = declaration->GetVariable(parameter->GetName());
			auto value = builder->AllocateRegister<T>(parameter->GetName());

			auto block = new PTX::BlockStatement();
			builder->AddStatement(block);
			builder->OpenScope(block);

			auto temp8 = builder->AllocateRegister<PTX::Int8Type, ResourceKind::Internal>(parameter->GetName());
			auto address = AddressGenerator<B>::template Generate<PTX::Int8Type>(variable, builder);
			builder->AddStatement(new PTX::LoadInstruction<B, PTX::Int8Type, PTX::GlobalSpace>(temp8, address));

			auto temp16 = builder->AllocateRegister<PTX::Int16Type, ResourceKind::Internal>(parameter->GetName());
			builder->AddStatement(new PTX::ConvertInstruction<PTX::Int16Type, PTX::Int8Type>(temp16, temp8));
			builder->AddStatement(new PTX::SetPredicateInstruction<PTX::Int16Type>(value, temp16, new PTX::Value<PTX::Int16Type>(0), PTX::Int16Type::ComparisonOperator::NotEqual));

			builder->CloseScope();
		}
		else
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
