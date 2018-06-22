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

template<PTX::Bits B>
class ParameterGenerator
{
public:
	using NodeType = HorseIR::Parameter;

	template<class T>
	static void Generate(HorseIR::Parameter *parameter, Builder *builder)
	{
		auto function = builder->GetCurrentFunction();
		auto resources = builder->GetCurrentResources();

		std::string variableName = function->GetName() + "_" + parameter->GetName();
		auto declaration = new PTX::PointerDeclaration<T, B>(variableName);
		function->AddParameter(declaration);
		auto variable = declaration->GetVariable(variableName);

		auto block = new PTX::BlockStatement();
		builder->AddStatement(block);
		builder->OpenScope(block);

		auto address = AddressGenerator<B>::template Generate<T>(variable, builder);
		auto value = resources->template AllocateRegister<T>(parameter->GetName());
		builder->AddStatement(new PTX::LoadInstruction<B, T, PTX::GlobalSpace>(value, address));

		builder->CloseScope();
	}
};
