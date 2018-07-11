#pragma once

#include "Codegen/Builder.h"
#include "Codegen/ResourceAllocator.h"
#include "Codegen/Generators/AddressGenerator.h"

#include "HorseIR/Tree/Statements/ReturnStatement.h"

#include "PTX/Type.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Instructions/Comparison/SelectInstruction.h"
#include "PTX/Instructions/Data/ConvertInstruction.h"
#include "PTX/Instructions/Data/StoreInstruction.h"
#include "PTX/Instructions/ControlFlow/ReturnInstruction.h"
#include "PTX/Statements/BlockStatement.h"

namespace Codegen {

template<PTX::Bits B>
class ReturnGenerator
{
public:
	using NodeType = HorseIR::ReturnStatement;

	template<class T>
	static void Generate(HorseIR::ReturnStatement *ret, Builder *builder)
	{
		//TODO: Specialize?
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			const std::string returnName = "$return";
			auto declaration = new PTX::PointerDeclaration<B, PTX::Int8Type>(returnName);
			builder->AddParameter(declaration);
			auto variable = declaration->GetVariable(returnName);

			auto block = new PTX::BlockStatement();
			builder->AddStatement(block);
			builder->OpenScope(block);

			auto address = AddressGenerator<B>::template Generate<PTX::Int8Type>(variable, builder);
			auto value = builder->GetRegister<PTX::PredicateType>(ret->GetVariableName());

			auto temp32 = builder->AllocateRegister<PTX::Int32Type, ResourceKind::Internal>(ret->GetVariableName());
			auto temp8 = builder->AllocateRegister<PTX::Int8Type, ResourceKind::Internal>(ret->GetVariableName());

			builder->AddStatement(new PTX::SelectInstruction<PTX::Int32Type>(temp32, new PTX::Value<PTX::Int32Type>(1), new PTX::Value<PTX::Int32Type>(0), value));
			builder->AddStatement(new PTX::ConvertInstruction<PTX::Int8Type, PTX::Int32Type>(temp8, temp32));
			builder->AddStatement(new PTX::StoreInstruction<B, PTX::Int8Type, PTX::GlobalSpace>(address, temp8));
			builder->AddStatement(new PTX::ReturnInstruction());

			builder->CloseScope();
		}
		else
		{
			const std::string returnName = "$return";
			auto declaration = new PTX::PointerDeclaration<B, T>(returnName);
			builder->AddParameter(declaration);
			auto variable = declaration->GetVariable(returnName);

			auto block = new PTX::BlockStatement();
			builder->AddStatement(block);
			builder->OpenScope(block);

			auto address = AddressGenerator<B>::template Generate<T>(variable, builder);
			auto value = builder->GetRegister<T>(ret->GetVariableName());
			builder->AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(address, value));
			builder->AddStatement(new PTX::ReturnInstruction());

			builder->CloseScope();
		}
	}
};

}
