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
			OperandGenerator<B, PTX::PredicateType> opGen(this->m_builder);
			auto value = opGen.GenerateRegister(ret->GetIdentifier());
			auto converted = ConversionGenerator::ConvertSource<PTX::Int8Type>(this->m_builder, value);

			Generate(converted, indexKind);
		}
		else
		{
			OperandGenerator<B, T> opGen(this->m_builder);
			auto value = opGen.GenerateRegister(ret->GetIdentifier());
			Generate(value, indexKind);
		}
	}

	template<class T, typename Enable = std::enable_if_t<PTX::StoreInstruction<B, T, PTX::GlobalSpace, PTX::StoreSynchronization::Weak, false>::TypeSupported>>
	void Generate(const PTX::Register<T> *value, IndexKind indexKind)
	{
		const std::string returnName = "$return";
		auto declaration = new PTX::PointerDeclaration<B, T>(returnName);
		this->m_builder->AddParameter(declaration);
		auto variable = declaration->GetVariable(returnName);

		AddressGenerator<B> addressGenerator(this->m_builder);
		auto address = addressGenerator.template GenerateParameter<T, PTX::GlobalSpace>(variable, indexKind);

		this->m_builder->AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(address, value));
		this->m_builder->AddStatement(new PTX::ReturnInstruction());
	}
};

}
