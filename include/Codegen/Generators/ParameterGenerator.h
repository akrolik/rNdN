#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/ResourceAllocator.h"
#include "Codegen/Generators/AddressGenerator.h"
#include "Codegen/Generators/Expressions/ConversionGenerator.h"

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

	using IndexKind = typename AddressGenerator<B>::IndexKind;

	template<class T>
	void Generate(const HorseIR::Parameter *parameter, IndexKind indexKind)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			auto value = this->m_builder->AllocateTemporary<PTX::Int8Type>();
			auto converted = this->m_builder->AllocateRegister<PTX::PredicateType>(parameter->GetName());

			Generate(parameter->GetName(), value, indexKind);
			ConversionGenerator::ConvertSource(this->m_builder, converted, value);
		}
		else
		{
			auto destination = this->m_builder->AllocateRegister<T>(parameter->GetName());
			Generate(parameter->GetName(), destination, indexKind);
		}
	}

	template<class T>
	void Generate(const std::string& name, const PTX::Register<T> *destination, IndexKind indexKind)
	{
		auto declaration = new PTX::PointerDeclaration<B, T>(name);
		this->m_builder->AddParameter(declaration);
		auto variable = declaration->GetVariable(name);

		AddressGenerator<B> addressGenerator(this->m_builder);
		auto address = addressGenerator.template GenerateParameter<T, PTX::GlobalSpace>(variable, indexKind);

		this->m_builder->AddStatement(new PTX::LoadInstruction<B, T, PTX::GlobalSpace>(destination, address));
	}
};

}
