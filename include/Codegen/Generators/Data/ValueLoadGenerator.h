#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"
#include "Codegen/Generators/AddressGenerator.h"
#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/Expressions/ConversionGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class ValueLoadGenerator : public Generator
{
public:
	using Generator::Generator;

	template<class T>
	const PTX::Register<T> *GeneratePointer(const std::string& name, const PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		return GeneratePointer<T>(name, index, name);
	}

	template<class T>
	const PTX::Register<T> *GeneratePointer(const std::string& name, const PTX::TypedOperand<PTX::UInt32Type> *index, const std::string& destinationName)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Allocate a register and load the value according to the index

		auto destination = resources->AllocateRegister<T>(destinationName);
		GeneratePointer(name, destination, index);
		return destination;
	}

	template<class T>
	void GeneratePointer(const std::string& name, const PTX::Register<T> *destination, const PTX::TypedOperand<PTX::UInt32Type> *index)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Predicate (1-bit) values are stored as signed 8-bit integers on the CPU side. Loading thus requires conversion

		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			auto temp = resources->AllocateTemporary<PTX::Int8Type>();
			GeneratePointer(name, temp, index);
			ConversionGenerator::ConvertSource(this->m_builder, destination, temp);
		}
		else
		{
			// Get the address register for the parameter

			auto addressRegister = resources->GetRegister<PTX::UIntType<B>>(NameUtils::DataAddressName(name));

			// Generate the address for the correct index

			AddressGenerator<B> addressGenerator(this->m_builder);
			auto address = addressGenerator.template GenerateAddress<T, PTX::GlobalSpace>(addressRegister, index);

			// Load the value from the fetched address

			this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::GlobalSpace>(destination, address));
		}
	}

	template<class T>
	const PTX::Register<T> *GenerateConstant(const std::string& name)
	{
		// Get the address for the constant parameter

		auto resources = this->m_builder.GetLocalResources();
		auto kernelResources = this->m_builder.GetKernelResources();

		auto destination = resources->AllocateRegister<T>(name);

		auto variable = kernelResources->template GetParameter<T, PTX::ParameterSpace>(name);
		auto address = new PTX::MemoryAddress<B, T, PTX::ParameterSpace>(variable);

		// Load the value from the fetched address

		this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::ParameterSpace>(destination, address));

		return destination;
	}
};

}
