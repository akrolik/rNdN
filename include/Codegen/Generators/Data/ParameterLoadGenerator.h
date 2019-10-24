#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"
#include "Codegen/Generators/AddressGenerator.h"
#include "Codegen/Generators/IndexGenerator.h"
#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/Data/ValueLoadGenerator.h"
#include "Codegen/Generators/Expressions/ConversionGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

#include "Runtime/RuntimeUtils.h"

namespace Codegen {

template<PTX::Bits B>
class ParameterLoadGenerator : public Generator
{
public:
	using Generator::Generator;

	void Generate(const HorseIR::Parameter *parameter)
	{
		this->m_builder.AddStatement(new PTX::CommentStatement(HorseIR::PrettyPrinter::PrettyString(parameter, true)));

		auto& inputOptions = this->m_builder.GetInputOptions();
		auto shape = inputOptions.ParameterShapes.at(parameter->GetSymbol());

		DispatchType(*this, parameter->GetType(), parameter->GetName(), shape);
	}

	void Generate(const std::vector<HorseIR::Type *>& returnTypes)
	{
		auto& inputOptions = this->m_builder.GetInputOptions();

		auto returnIndex = 0u;
		for (const auto& returnType : returnTypes)
		{
			auto name = NameUtils::ReturnName(returnIndex);
			this->m_builder.AddStatement(new PTX::CommentStatement(name + ":" + HorseIR::PrettyPrinter::PrettyString(returnType, true)));

			auto shape = inputOptions.ReturnShapes.at(returnIndex);
			DispatchType(*this, returnType, name, shape, true);

			returnIndex++;
		}
	}

	template<class T>
	void Generate(const std::string& name, const Analysis::Shape *shape, bool returnParameter = false)
	{
		// Predicate (1-bit) values are stored as signed 8-bit integers on the CPU side. Loading thus requires conversion

		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			Generate<PTX::Int8Type>(name, shape);
		}
		else
		{
			auto& inputOptions = this->m_builder.GetInputOptions();
			bool dynamicSize = (Runtime::RuntimeUtils::IsDynamicDataShape(shape, inputOptions.ThreadGeometry) && !returnParameter);

			if (Analysis::ShapeUtils::IsShape<Analysis::VectorShape>(shape))
			{
				GenerateVector<T>(name, dynamicSize);
			}
			else if (Analysis::ShapeUtils::IsShape<Analysis::ListShape>(shape))
			{
				GenerateList<T>(name, dynamicSize);
			}
		}
	}

	template<class T>
	void GenerateVector(const std::string& name, bool dynamicSize)
	{
		// Get the variable from the kernel resources, and generate the address

		auto kernelResources = this->m_builder.GetKernelResources();
		auto variable = kernelResources->template GetParameter<PTX::PointerType<B, T>, PTX::ParameterSpace>(name);

		AddressGenerator<B> addressGenerator(this->m_builder);
		addressGenerator.template LoadParameterAddress<T>(variable);

		// Load the dynamic size parameter if needed

		if (dynamicSize)
		{
			ValueLoadGenerator<B> loadGenerator(this->m_builder);
			loadGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::SizeName(name));
		}
	}

	template<class T>
	void GenerateList(const std::string& name, bool dynamicSize)
	{
		// Get the variable from the kernel resources, and generate the address. This will load the cell contents in addition to the list structure

		auto kernelResources = this->m_builder.GetKernelResources();
		auto variable = kernelResources->template GetParameter<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>, PTX::ParameterSpace>(name);

		IndexGenerator indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateCellIndex();

		AddressGenerator<B> addressGenerator(this->m_builder);
		addressGenerator.template LoadParameterAddress<T>(variable, index);

		// Load the dynamic size parameter if needed

		if (dynamicSize)
		{
			auto sizeName = NameUtils::SizeName(name);
			auto sizeVariable = kernelResources->template GetParameter<PTX::PointerType<B, PTX::UInt32Type>, PTX::ParameterSpace>(sizeName);

			AddressGenerator<B> addressGenerator(this->m_builder);
			addressGenerator.template LoadParameterAddress<PTX::UInt32Type>(sizeVariable);

			ValueLoadGenerator<B> loadGenerator(this->m_builder);
			loadGenerator.template GeneratePointer<PTX::UInt32Type>(sizeName, index);
		}
	}
};

}
