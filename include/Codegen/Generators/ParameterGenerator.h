#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/AddressGenerator.h"
#include "Codegen/Generators/Expressions/ConversionGenerator.h"
#include "Codegen/Resources/RegisterAllocator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class ParameterGenerator : public Generator
{
public:
	using Generator::Generator;

	using IndexKind = typename AddressGenerator<B>::IndexKind;

	template<class T>
	void Generate(const HorseIR::Parameter *parameter)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto indexKind = GetIndexKind(parameter);

		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			auto value = resources->AllocateTemporary<PTX::Int8Type>();
			auto converted = resources->AllocateRegister<PTX::PredicateType>(parameter->GetName());

			Generate(parameter->GetName(), value, indexKind);
			ConversionGenerator::ConvertSource(this->m_builder, converted, value);
		}
		else
		{
			auto destination = resources->AllocateRegister<T>(parameter->GetName());
			Generate(parameter->GetName(), destination, indexKind);
		}
	}

	template<class T>
	void Generate(const std::string& name, const PTX::Register<T> *destination, IndexKind indexKind)
	{
		auto declaration = new PTX::PointerDeclaration<B, T>(name);
		this->m_builder.AddParameter(name, declaration);
		auto variable = declaration->GetVariable(name);

		AddressGenerator<B> addressGenerator(this->m_builder);
		auto address = addressGenerator.template GenerateParameter<T, PTX::GlobalSpace>(variable, indexKind);

		this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::GlobalSpace>(destination, address));
	}

	IndexKind GetIndexKind(const HorseIR::Parameter *parameter) const
	{
		auto& inputOptions = this->m_builder.GetInputOptions();

		// Translate the high-level indexing to a low-level thread mapping

		auto indexKind = inputOptions.ParameterIndexKinds.at(parameter);
		switch (indexKind)
		{
			case InputOptions::IndexKind::Scalar:
				return IndexKind::Null;
			case InputOptions::IndexKind::Vector:
				return IndexKind::Global;
			case InputOptions::IndexKind::List:
			case InputOptions::IndexKind::Cell:
				//TODO: List indexing
				break;
		}

		Utils::Logger::LogError("Unknown thread mapping for index kind " + InputOptions::IndexKindString(indexKind));
	}
};

}
