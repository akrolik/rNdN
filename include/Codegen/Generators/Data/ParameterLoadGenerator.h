#pragma once

#include "Codegen/Generators/Generator.h"

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"
#include "Codegen/Generators/AddressGenerator.h"
#include "Codegen/Generators/IndexGenerator.h"
#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/Data/ValueLoadGenerator.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

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
			Generate<PTX::Int8Type>(name, shape, returnParameter);
		}
		else
		{
			auto& inputOptions = this->m_builder.GetInputOptions();
			auto loadSize = Runtime::RuntimeUtils::IsDynamicDataShape(shape, inputOptions.ThreadGeometry);

			auto kernelResources = this->m_builder.GetKernelResources();
			if (Analysis::ShapeUtils::IsShape<Analysis::VectorShape>(shape))
			{
				auto parameter = kernelResources->template GetParameter<PTX::PointerType<B, T>>(name);
				GenerateVector<T>(parameter, loadSize, returnParameter);
			}
			else if (Analysis::ShapeUtils::IsShape<Analysis::ListShape>(shape))
			{
				auto parameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>>(name);
				GenerateList<T>(parameter, loadSize, returnParameter);
			}
		}
	}

	template<class T>
	void GenerateVector(const PTX::ParameterVariable<PTX::PointerType<B, T>> *parameter, bool loadSize = false, bool returnParameter = false)
	{
		// Get the variable from the kernel resources, and generate the address

		AddressGenerator<B> addressGenerator(this->m_builder);
		addressGenerator.template LoadParameterAddress<T>(parameter);

		// Load the dynamic size parameter if needed

		if (loadSize)
		{
			auto kernelResources = this->m_builder.GetKernelResources();
			auto sizeName = NameUtils::SizeName(parameter);

			ValueLoadGenerator<B> loadGenerator(this->m_builder);
			if (returnParameter)
			{
				// Return dynamic shapes use pointer types for accumulating size

				auto sizeParameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::UInt32Type>>(sizeName);

				AddressGenerator<B> addressGenerator(this->m_builder);
				addressGenerator.template LoadParameterAddress<PTX::UInt32Type>(sizeParameter);
				loadGenerator.template GeneratePointer<PTX::UInt32Type>(sizeParameter);
			}
			else
			{
				auto sizeParameter = kernelResources->template GetParameter<PTX::UInt32Type>(sizeName);
				loadGenerator.template GenerateConstant<PTX::UInt32Type>(sizeParameter);
			}
		}
	}

	template<class T>
	void GenerateList(const PTX::ParameterVariable<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>> *parameter, bool loadSize = false, bool returnParameter = false)
	{
		// Get the variable from the kernel resources, and generate the address. This will load the cell contents in addition to the list structure

		IndexGenerator indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateCellIndex();

		AddressGenerator<B> addressGenerator(this->m_builder);
		addressGenerator.template LoadParameterAddress<T>(parameter, index);

		// Load the dynamic size parameter if needed

		if (loadSize)
		{
			auto kernelResources = this->m_builder.GetKernelResources();
			auto sizeName = NameUtils::SizeName(parameter);

			auto sizeParameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::UInt32Type>>(sizeName);

			AddressGenerator<B> addressGenerator(this->m_builder);
			addressGenerator.template LoadParameterAddress<PTX::UInt32Type>(sizeParameter);

			ValueLoadGenerator<B> loadGenerator(this->m_builder);
			loadGenerator.template GeneratePointer<PTX::UInt32Type>(sizeParameter, index);
		}
	}
};

}
