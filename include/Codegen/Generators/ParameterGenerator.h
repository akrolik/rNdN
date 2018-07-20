#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/ResourceAllocator.h"
#include "Codegen/Generators/AddressGenerator.h"

#include "HorseIR/Tree/Parameter.h"

#include "PTX/Type.h"
#include "PTX/Declarations/VariableDeclaration.h"
#include "PTX/Instructions/Comparison/SetPredicateInstruction.h"
#include "PTX/Instructions/Data/ConvertInstruction.h"
#include "PTX/Instructions/Data/LoadInstruction.h"

namespace Codegen {

template<PTX::Bits B>
class ParameterGenerator : public Generator
{
public:
	using Generator::Generator;

	template<class T>
	void Generate(const HorseIR::Parameter *parameter)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			auto declaration = new PTX::PointerDeclaration<B, PTX::Int8Type>(parameter->GetName());
			this->m_builder->AddParameter(declaration);
			auto variable = declaration->GetVariable(parameter->GetName());
			auto value = this->m_builder->AllocateRegister<T>(parameter->GetName());

			auto temp8 = this->m_builder->AllocateTemporary<PTX::Int8Type>();
			auto temp16 = this->m_builder->AllocateTemporary<PTX::Int16Type>();

			AddressGenerator<B> addressGenerator(this->m_builder);
			auto address = addressGenerator.template GenerateParameter<PTX::Int8Type, PTX::GlobalSpace>(variable);

			this->m_builder->AddStatement(new PTX::LoadInstruction<B, PTX::Int8Type, PTX::GlobalSpace>(temp8, address));
			this->m_builder->AddStatement(new PTX::ConvertInstruction<PTX::Int16Type, PTX::Int8Type>(temp16, temp8));
			this->m_builder->AddStatement(new PTX::SetPredicateInstruction<PTX::Int16Type>(value, temp16, new PTX::Value<PTX::Int16Type>(0), PTX::Int16Type::ComparisonOperator::NotEqual));
		}
		else
		{
			auto declaration = new PTX::PointerDeclaration<B, T>(parameter->GetName());
			this->m_builder->AddParameter(declaration);
			auto variable = declaration->GetVariable(parameter->GetName());
			auto value = this->m_builder->AllocateRegister<T>(parameter->GetName());

			AddressGenerator<B> addressGenerator(this->m_builder);
			auto address = addressGenerator.template GenerateParameter<T, PTX::GlobalSpace>(variable);

			this->m_builder->AddStatement(new PTX::LoadInstruction<B, T, PTX::GlobalSpace>(value, address));
		}
	}
};

}
