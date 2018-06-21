#pragma once

#include "Codegen/GeneratorState.h"
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
	static void Generate(HorseIR::ReturnStatement *ret, GeneratorState *state)
	{
		auto function = state->GetCurrentFunction();
		auto resources = state->GetCurrentResources();

		std::string variableName = function->GetName() + "_return";
		auto declaration = new PTX::PointerDeclaration<T, B>(variableName);
		function->AddParameter(declaration);
		auto variable = declaration->GetVariable(variableName);

		auto block = new PTX::BlockStatement();
		ResourceAllocator *localResources = state->OpenScope(block);

		auto address = AddressGenerator<B>::template Generate<T>(variable, state);
		auto value = resources->template AllocateRegister<T>(ret->GetIdentifier());
		block->AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(address, value));
		block->AddStatement(new PTX::ReturnInstruction());

		state->CloseScope();
		function->AddStatement(block);
	}
};
