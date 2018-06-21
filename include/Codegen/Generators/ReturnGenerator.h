#pragma once

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
	static void Generate(HorseIR::ReturnStatement *ret, PTX::DataFunction<PTX::VoidType> *currentFunction, ResourceAllocator *resources)
	{
		std::string variableName = currentFunction->GetName() + "_return";
		auto declaration = new PTX::PointerDeclaration<T, B>(variableName);
		currentFunction->AddParameter(declaration);
		auto variable = declaration->GetVariable(variableName);

		auto block = new PTX::BlockStatement();
		ResourceAllocator *localResources = new ResourceAllocator();

		auto address = AddressGenerator<B>::template Generate<T>(variable, block, localResources);
		auto value = resources->template AllocateRegister<T>(ret->GetIdentifier());
		block->AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(address, value));
		block->AddStatement(new PTX::ReturnInstruction());
		block->InsertStatements(localResources->GetRegisterDeclarations(), 0);

		currentFunction->AddStatement(block);
	}
};
