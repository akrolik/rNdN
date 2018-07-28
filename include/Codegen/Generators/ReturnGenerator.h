#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/ResourceAllocator.h"
#include "Codegen/Generators/AddressGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "HorseIR/Tree/Statements/ReturnStatement.h"

#include "PTX/Type.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Instructions/Comparison/SelectInstruction.h"
#include "PTX/Instructions/Data/ConvertInstruction.h"
#include "PTX/Instructions/Data/StoreInstruction.h"
#include "PTX/Instructions/ControlFlow/ReturnInstruction.h"

namespace Codegen {

template<PTX::Bits B>
class ReturnGenerator : public Generator
{
public:
	using Generator::Generator;

	using IndexKind = typename AddressGenerator<B>::IndexKind;

	template<class T>
	void Generate(const HorseIR::ReturnStatement *ret, IndexKind indexKind)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			const std::string returnName = "$return";
			auto declaration = new PTX::PointerDeclaration<B, PTX::Int8Type>(returnName);
			this->m_builder->AddParameter(declaration);
			auto variable = declaration->GetVariable(returnName);

			AddressGenerator<B> addressGenerator(this->m_builder);
			auto address = addressGenerator.template GenerateParameter<PTX::Int8Type, PTX::GlobalSpace>(variable, indexKind);

			OperandGenerator<B, PTX::PredicateType> opGen(this->m_builder);
			auto value = opGen.GenerateRegister(ret->GetIdentifier());

			auto temp32 = this->m_builder->AllocateTemporary<PTX::Int32Type>();
			auto temp8 = this->m_builder->AllocateTemporary<PTX::Int8Type>();

			this->m_builder->AddStatement(new PTX::SelectInstruction<PTX::Int32Type>(temp32, new PTX::Value<PTX::Int32Type>(1), new PTX::Value<PTX::Int32Type>(0), value));
			this->m_builder->AddStatement(new PTX::ConvertInstruction<PTX::Int8Type, PTX::Int32Type>(temp8, temp32));
			this->m_builder->AddStatement(new PTX::StoreInstruction<B, PTX::Int8Type, PTX::GlobalSpace>(address, temp8));
			this->m_builder->AddStatement(new PTX::ReturnInstruction());
		}
		else
		{
			const std::string returnName = "$return";
			auto declaration = new PTX::PointerDeclaration<B, T>(returnName);
			this->m_builder->AddParameter(declaration);
			auto variable = declaration->GetVariable(returnName);

			AddressGenerator<B> addressGenerator(this->m_builder);
			auto address = addressGenerator.template GenerateParameter<T, PTX::GlobalSpace>(variable, indexKind);

			OperandGenerator<B, T> opGen(this->m_builder);
			auto value = opGen.GenerateRegister(ret->GetIdentifier());

			this->m_builder->AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(address, value));
			this->m_builder->AddStatement(new PTX::ReturnInstruction());
		}
	}
};

}
