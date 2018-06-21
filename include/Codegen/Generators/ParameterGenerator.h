#pragma once

#include "Codegen/GeneratorState.h"
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
	static void Generate(HorseIR::Parameter *parameter, GeneratorState *state)
	{
		auto function = state->GetCurrentFunction();
		auto resources = state->GetCurrentResources();

		std::string variableName = function->GetName() + "_" + parameter->GetName();
		auto declaration = new PTX::PointerDeclaration<T, B>(variableName);
		function->AddParameter(declaration);
		auto variable = declaration->GetVariable(variableName);

		auto block = new PTX::BlockStatement();
		ResourceAllocator *localResources = state->OpenScope(block);

		auto address = AddressGenerator<B>::template Generate<T>(variable, state);
		auto value = resources->template AllocateRegister<T>(parameter->GetName());
		block->AddStatement(new PTX::LoadInstruction<B, T, PTX::GlobalSpace>(value, address));

		state->CloseScope();
		function->AddStatement(block);
	}
};
