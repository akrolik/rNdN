#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"
#include "Codegen/Generators/AddressGenerator.h"
#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/Expressions/ConversionGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

#include "Utils/Logger.h"

namespace Codegen {

template<PTX::Bits B>
class ValueLoadGenerator : public Generator
{
public:
	using Generator::Generator;

	template<class T>
	const PTX::Register<T> *GenerateParameter(const HorseIR::Parameter *parameter, const PTX::TypedOperand<PTX::UInt32Type> *dataIndex, const std::string& sourceName = "")
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			// Boolean parameters are stored as 8-bit integers

			auto value = GenerateParameter<PTX::Int8Type>(parameter, dataIndex, sourceName);
			return ConversionGenerator::ConvertSource<T, PTX::Int8Type>(this->m_builder, value);
		}
		else
		{
			auto kernelResources = this->m_builder.GetKernelResources();
			auto& inputOptions = this->m_builder.GetInputOptions();

			auto shape = inputOptions.ParameterShapes.at(parameter);
			auto name = NameUtils::VariableName(parameter);
			if (Analysis::ShapeUtils::IsShape<Analysis::VectorShape>(shape))
			{
				auto parameter = kernelResources->template GetParameter<PTX::PointerType<B, T>>(name);
				return GeneratePointer<T>(parameter, dataIndex, sourceName);
			}
			else if (Analysis::ShapeUtils::IsShape<Analysis::ListShape>(shape))
			{
				auto parameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>>(name);
				return GeneratePointer<T>(parameter, dataIndex, sourceName);
			}

			Utils::Logger::LogError("Unable to generate load for parameter " + HorseIR::PrettyPrinter::PrettyString(parameter) + " with shape " + Analysis::ShapeUtils::ShapeString(shape));
		}
	}

	template<class T>
	const PTX::Register<T> *GeneratePointer(const PTX::ParameterVariable<PTX::PointerType<B, T>> *parameter, const PTX::TypedOperand<PTX::UInt32Type> *index = nullptr, const std::string& destinationName = "")
	{
		// Get the address and destination name, then load the value

		auto resources = this->m_builder.GetLocalResources();
		auto addressRegister = resources->GetRegister<PTX::UIntType<B>>(NameUtils::DataAddressName(parameter));

		auto name = (destinationName == "") ? parameter->GetName() : destinationName;
		return GeneratePointer<T>(addressRegister, index, name);
	}

	template<class T>
	const PTX::Register<T> *GeneratePointer(const PTX::ParameterVariable<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>> *parameter, const PTX::TypedOperand<PTX::UInt32Type> *index = nullptr, const std::string& destinationName = "")
	{
		// Get the address and destination name, then load the value

		auto resources = this->m_builder.GetLocalResources();
		auto addressRegister = resources->GetRegister<PTX::UIntType<B>>(NameUtils::DataAddressName(parameter));

		auto name = (destinationName == "") ? parameter->GetName() : destinationName;
		return GeneratePointer<T>(addressRegister, index, name);
	}

	template<class T>
	const PTX::Register<T> *GeneratePointer(const PTX::Register<PTX::UIntType<B>> *addressRegister, const PTX::TypedOperand<PTX::UInt32Type> *index, const std::string& destinationName)
	{
		// Get the address register for the parameter

		auto resources = this->m_builder.GetLocalResources();
		auto destination = resources->AllocateRegister<T>(destinationName);

		// Generate the address for the correct index

		AddressGenerator<B> addressGenerator(this->m_builder);
		auto address = addressGenerator.template GenerateAddress<T, PTX::GlobalSpace>(addressRegister, index);

		// Load the value from the fetched address

		this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::GlobalSpace>(destination, address));

		return destination;
	}

	template<class T>
	const PTX::Register<T> *GenerateConstant(const PTX::ParameterVariable<T> *parameter)
	{
		// Get the address for the constant parameter

		auto resources = this->m_builder.GetLocalResources();
		auto destination = resources->AllocateRegister<T>(parameter->GetName());

		// Load the value from the fetched address

		auto address = new PTX::MemoryAddress<B, T, PTX::ParameterSpace>(parameter);
		this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::ParameterSpace>(destination, address));

		return destination;
	}
};

}
