#pragma once

#include "Codegen/Builder.h"
#include "Codegen/ResourceAllocator.h"
#include "Codegen/Generators/AddressGenerator.h"

#include "HorseIR/Tree/Statements/ReturnStatement.h"

#include "PTX/Type.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Functions/DataFunction.h"
#include "PTX/Instructions/Data/StoreInstruction.h"
#include "PTX/Instructions/ControlFlow/ReturnInstruction.h"
#include "PTX/Statements/BlockStatement.h"

template<PTX::Bits B>
class ReturnGenerator
{
public:
	using NodeType = HorseIR::ReturnStatement;

	template<class T>
	static void Generate(HorseIR::ReturnStatement *ret, Builder *builder)
	{
		auto function = builder->GetCurrentFunction();
		auto resources = builder->GetCurrentResources();

		std::string variableName = function->GetName() + "_return";
		auto declaration = new PTX::PointerDeclaration<T, B>(variableName);
		function->AddParameter(declaration);
		auto variable = declaration->GetVariable(variableName);

		auto block = new PTX::BlockStatement();
		builder->AddStatement(block);
		builder->OpenScope(block);

		auto address = AddressGenerator<B>::template Generate<T>(variable, builder);
		auto value = resources->template AllocateRegister<T>(ret->GetVariableName());
		builder->AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(address, value));
		builder->AddStatement(new PTX::ReturnInstruction());

		builder->CloseScope();
	}
};
