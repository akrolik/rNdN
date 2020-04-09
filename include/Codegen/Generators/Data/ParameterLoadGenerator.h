#pragma once

#include "Codegen/Generators/Generator.h"

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"
#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/Data/ValueLoadGenerator.h"
#include "Codegen/Generators/Indexing/AddressGenerator.h"
#include "Codegen/Generators/Indexing/DataIndexGenerator.h"

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

	std::string Name() const override { return "ParameterLoadGenerator"; }

	void Generate(const HorseIR::Function *function)
	{
		auto& inputOptions = this->m_builder.GetInputOptions();

		for (const auto& parameter : function->GetParameters())
		{
			this->m_builder.AddStatement(new PTX::CommentStatement(HorseIR::PrettyPrinter::PrettyString(parameter, true)));

			auto shape = inputOptions.ParameterShapes.at(parameter);
			DispatchType(*this, parameter->GetType(), parameter->GetName(), shape);
		}

		auto returnIndex = 0u;
		for (const auto& returnType : function->GetReturnTypes())
		{
			auto name = NameUtils::ReturnName(returnIndex);

			this->m_builder.AddStatement(new PTX::CommentStatement(name + ":" + HorseIR::PrettyPrinter::PrettyString(returnType, true)));

			auto shape = inputOptions.ReturnShapes.at(returnIndex);
			DispatchType(*this, returnType, name, shape, true);

			returnIndex++;
		}
	}

	template<class T>
	void GenerateVector(const std::string& name, const Analysis::Shape *shape, bool returnParameter = false)
	{
		// Predicate (1-bit) values are stored as signed 8-bit integers on the CPU side. Loading thus requires conversion

		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			GenerateVector<PTX::Int8Type>(name, shape, returnParameter);
		}
		else
		{
			auto kernelResources = this->m_builder.GetKernelResources();
			auto parameter = kernelResources->template GetParameter<PTX::PointerType<B, T>>(name);

			// Get the variable from the kernel resources, and generate the address

			GenerateParameterAddress<T>(parameter);

			// Load the dynamic size parameter if needed
						
			if (returnParameter)
			{
				// Return dynamic shapes use pointer types for accumulating size

				auto& inputOptions = this->m_builder.GetInputOptions();
				if (Runtime::RuntimeUtils::IsDynamicReturnShape(shape, inputOptions.ThreadGeometry))
				{
					GeneratePointerSize(parameter);
				}
			}
			else
			{
				GenerateConstantSize(parameter);
			}
		}
	}

	template<class T>
	void GenerateList(const std::string& name, const Analysis::Shape *shape, bool returnParameter = false)
	{
		// Predicate (1-bit) values are stored as signed 8-bit integers on the CPU side. Loading thus requires conversion

		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			GenerateList<PTX::Int8Type>(name, shape, returnParameter);
		}
		else
		{
			auto resources = this->m_builder.GetLocalResources();
			auto kernelResources = this->m_builder.GetKernelResources();

			// Get the variable from the kernel resources, and generate the address. This will load the cell contents in addition to the list structure

			using DataType = PTX::PointerType<B, T, PTX::GlobalSpace>;

			auto parameter = kernelResources->template GetParameter<PTX::PointerType<B, DataType>>(name);
			auto parameterAddress = GenerateParameterAddress<DataType>(parameter);

			// Load the list data depending on the geometry (either list-in-list, or list-in-vector)

			auto& inputOptions = this->m_builder.GetInputOptions();
			if (inputOptions.IsVectorGeometry())
			{
				// Check cell count is constant

				auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape);
				if (auto cellSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(listShape->GetListSize()))
				{
					// Load each cell into a separate address register

					for (auto index = 0u; index < cellSize->GetValue(); ++index)
					{
						// Load the address of the cell

						auto cellAddressName = NameUtils::DataCellAddressName(parameter, index);
						auto cellAddress = resources->template AllocateRegister<PTX::UIntType<B>>(cellAddressName);

						auto dataPointer = new PTX::PointerRegisterAdapter<B, T, PTX::GlobalSpace>(cellAddress);
						auto indexedAddress = new PTX::RegisterAddress<B, DataType, PTX::GlobalSpace>(parameterAddress, index);

						this->m_builder.AddStatement(new PTX::LoadInstruction<B, DataType, PTX::GlobalSpace>(dataPointer, indexedAddress));

						// Load the dynamic size if needed

						if (!returnParameter || Runtime::RuntimeUtils::IsDynamicReturnShape(shape, inputOptions.ThreadGeometry))
						{
							GeneratePointerSize(parameter, index);
						}
					}
				}
				else
				{
					Error(name, shape, returnParameter);
				}
			}
			else if (inputOptions.IsListGeometry())
			{
				// Load the list depending on the respective cell threads

				DataIndexGenerator<B> indexGenerator(this->m_builder);
				auto index = indexGenerator.GenerateListIndex();

				// Get the address of the value in the indirection structure (by the bitsize of the address)

				auto globalIndexed = resources->template AllocateTemporary<PTX::UIntType<B>>();
				auto globalIndexedPointer = new PTX::PointerRegisterAdapter<B, DataType, PTX::GlobalSpace>(globalIndexed);

				AddressGenerator<B, DataType> addressGenerator(this->m_builder);
				auto offset = addressGenerator.template GenerateAddressOffset<B>(index);

				this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UIntType<B>>(globalIndexed, parameterAddress->GetVariable(), offset));

				// Load the address of the data

				auto cellAddressName = NameUtils::DataCellAddressName(parameter);
				auto cellRegister = resources->template AllocateRegister<PTX::UIntType<B>>(cellAddressName);

				auto dataPointer = new PTX::PointerRegisterAdapter<B, T, PTX::GlobalSpace>(cellRegister);
				auto indexedAddress = new PTX::RegisterAddress<B, DataType, PTX::GlobalSpace>(globalIndexedPointer);

				this->m_builder.AddStatement(new PTX::LoadInstruction<B, DataType, PTX::GlobalSpace>(dataPointer, indexedAddress));

				// Load the dynamic size parameter if needed

				if (!returnParameter || Runtime::RuntimeUtils::IsDynamicReturnShape(shape, inputOptions.ThreadGeometry))
				{
					GeneratePointerSize(parameter, index);
				}
			}
			else
			{
				Error(name, shape, returnParameter);
			}
		}
	}

	template<class T>
	void GenerateTuple(unsigned int index, const std::string& name, const Analysis::Shape *shape, bool returnParameter = false)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			// Predicate values are stored as 8-bit integers on the CPU

			GenerateTuple<PTX::Int8Type>(index, name, shape, returnParameter);
		}
		else
		{
			if (!this->m_builder.GetInputOptions().IsVectorGeometry())
			{
				Error(name, shape, returnParameter);
			}

			auto resources = this->m_builder.GetLocalResources();
			auto kernelResources = this->m_builder.GetKernelResources();

			using DataType = PTX::PointerType<B, T, PTX::GlobalSpace>;

			// If this is the first cell, load the main list structure
			
			auto parameter = kernelResources->template GetParameter<PTX::PointerType<B, DataType>>(name);

			if (index == 0)
			{
				GenerateParameterAddress<DataType>(parameter);
			}

			auto dataAddressName = NameUtils::DataAddressName(parameter);
			auto globalRegister = resources->template GetRegister<PTX::UIntType<B>>(dataAddressName);

			auto parameterAddress = new PTX::PointerRegisterAdapter<B, DataType, PTX::GlobalSpace>(globalRegister);

			// Load the contents of the cell

			auto cellAddressName = NameUtils::DataCellAddressName(parameter, index);
			auto cellAddress = resources->template AllocateRegister<PTX::UIntType<B>>(cellAddressName);

			auto dataPointer = new PTX::PointerRegisterAdapter<B, T, PTX::GlobalSpace>(cellAddress);
			auto indexedAddress = new PTX::RegisterAddress<B, DataType, PTX::GlobalSpace>(parameterAddress, index);

			this->m_builder.AddStatement(new PTX::LoadInstruction<B, DataType, PTX::GlobalSpace>(dataPointer, indexedAddress));

			// Load the dynamic size if needed

			auto& inputOptions = this->m_builder.GetInputOptions();
			if (!returnParameter || Runtime::RuntimeUtils::IsDynamicReturnShape(shape, inputOptions.ThreadGeometry))
			{
				GeneratePointerSize(parameter, index);
			}
		}
	}

	template<class T>
	void GeneratePointerSize(const PTX::ParameterVariable<PTX::PointerType<B, T>> *parameter, const PTX::TypedOperand<PTX::UInt32Type> *index = nullptr)
	{
		auto kernelResources = this->m_builder.GetKernelResources();
		auto sizeName = NameUtils::SizeName(parameter);

		auto sizeParameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::UInt32Type>>(sizeName);
		GenerateParameterAddress<PTX::UInt32Type>(sizeParameter);

		ValueLoadGenerator<B, PTX::UInt32Type> loadGenerator(this->m_builder);
		loadGenerator.GeneratePointer(sizeParameter, index);
	}

	template<class T>
	void GeneratePointerSize(const PTX::ParameterVariable<PTX::PointerType<B, T>> *parameter, unsigned int index)
	{
		auto kernelResources = this->m_builder.GetKernelResources();
		auto sizeName = NameUtils::SizeName(parameter);

		auto sizeParameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::UInt32Type>>(sizeName);
		GenerateParameterAddress<PTX::UInt32Type>(sizeParameter);

		auto cellSizeName = NameUtils::SizeName(parameter, index);

		ValueLoadGenerator<B, PTX::UInt32Type> loadGenerator(this->m_builder);
		loadGenerator.GeneratePointer(cellSizeName, sizeParameter, nullptr, index);
	}

	template<class T>
	void GenerateConstantSize(const PTX::ParameterVariable<T> *parameter)
	{
		auto kernelResources = this->m_builder.GetKernelResources();
		auto sizeName = NameUtils::SizeName(parameter);

		auto sizeParameter = kernelResources->template GetParameter<PTX::UInt32Type>(sizeName);

		ValueLoadGenerator<B, PTX::UInt32Type> loadGenerator(this->m_builder);
		loadGenerator.GenerateConstant(sizeParameter);
	}

	template<class T>
	const PTX::PointerRegisterAdapter<B, T, PTX::GlobalSpace> *GenerateParameterAddress(const PTX::ParameterVariable<PTX::PointerType<B, T>> *parameter)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Get the base address of the variable in generic space

		auto baseAddress = new PTX::MemoryAddress<B, PTX::PointerType<B, T>, PTX::ParameterSpace>(parameter);
		auto genericBase = new PTX::PointerRegisterAdapter<B, T>(resources->template AllocateTemporary<PTX::UIntType<B>>());

		this->m_builder.AddStatement(new PTX::LoadInstruction<B, PTX::PointerType<B, T>, PTX::ParameterSpace>(genericBase, baseAddress));

		// Convert the generic address of the underlying variable to the global space

		auto dataAddressName = NameUtils::DataAddressName(parameter);
		auto globalRegister = resources->template AllocateRegister<PTX::UIntType<B>>(dataAddressName);

		auto globalBase = new PTX::PointerRegisterAdapter<B, T, PTX::GlobalSpace>(globalRegister);
		auto genericAddress = new PTX::RegisterAddress<B, T>(genericBase);

		this->m_builder.AddStatement(new PTX::ConvertToAddressInstruction<B, T, PTX::GlobalSpace>(globalBase, genericAddress));

		return globalBase;
	}

	[[noreturn]] void Error(const std::string& name, const Analysis::Shape *shape, bool returnParameter = false)
	{
		if (returnParameter)
		{
			Generator::Error("return parameter load '" + name + "' for shape " + Analysis::ShapeUtils::ShapeString(shape));
		}
		Generator::Error("parameter load '" + name + "' for shape " + Analysis::ShapeUtils::ShapeString(shape));
	}
};

}
