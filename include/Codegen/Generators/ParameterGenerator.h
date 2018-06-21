#pragma once

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
	static void Generate(HorseIR::Parameter *parameter, PTX::DataFunction<PTX::VoidType> *currentFunction, ResourceAllocator *resources)
	{
		std::string variableName = currentFunction->GetName() + "_" + parameter->GetName();
		auto declaration = new PTX::PointerDeclaration<T, B>(variableName);
		currentFunction->AddParameter(declaration);
		auto variable = declaration->GetVariable(variableName);

		auto block = new PTX::BlockStatement();
		ResourceAllocator *localResources = new ResourceAllocator();

		auto address = AddressGenerator<B>::template Generate<T>(variable, block, localResources);
		auto value = resources->template AllocateRegister<T>(parameter->GetName());
		block->AddStatement(new PTX::LoadInstruction<B, T, PTX::GlobalSpace>(value, address));
		block->InsertStatements(localResources->GetRegisterDeclarations(), 0);

		currentFunction->AddStatement(block);
	}
};
