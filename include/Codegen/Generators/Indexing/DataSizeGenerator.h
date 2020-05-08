#pragma once

#include "Codegen/Generators/Generator.h"
#include "HorseIR/Traversal/ConstVisitor.h"

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"
#include "Codegen/Generators/TypeDispatch.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class DataSizeGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	using Generator::Generator;

	std::string Name() const override { return "DataSizeGenerator"; }

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateSize(const HorseIR::Parameter *parameter, unsigned int cellIndex = 0)
	{
		m_cellIndex = cellIndex;
		m_size = nullptr;

		DispatchType(*this, parameter->GetType(), parameter);
		if (m_size == nullptr)
		{
			Error("size for parameter " + HorseIR::PrettyPrinter::PrettyString(parameter));
		}
		return m_size;
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateSize(const HorseIR::Operand *operand)
	{
		m_size = nullptr;
		operand->Accept(*this);
		if (m_size == nullptr)
		{
			Error("size for operand " + HorseIR::PrettyPrinter::PrettyString(operand));
		}
		return m_size;
	}

	void Visit(const HorseIR::VectorLiteral *literal) override
	{
		m_size = new PTX::UInt32Value(literal->GetCount());
	}
	
	void Visit(const HorseIR::Identifier *identifier) override
	{
		// If the identifier is a parameter, then we can find the size

		auto& parameters = this->m_builder.GetInputOptions().Parameters;

		auto find = parameters.find(identifier->GetSymbol());
		if (find != parameters.end())
		{
			auto parameter = find->second;
			DispatchType(*this, parameter->GetType(), parameter);
		}
	}
	
	template<class T>
	void GenerateVector(const HorseIR::Parameter *parameter)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			GenerateVector<PTX::Int8Type>(parameter);
		}
		else
		{
			auto shape = this->m_builder.GetInputOptions().ParameterShapes.at(parameter);
			auto name = NameUtils::VariableName(parameter);

			auto kernelResources = this->m_builder.GetKernelResources();
			auto kernelParameter = kernelResources->template GetParameter<PTX::PointerType<B, T>>(name);

			m_size = GenerateSize(kernelParameter, shape);
		}
	}

	template<class T>
	void GenerateList(const HorseIR::Parameter *parameter)
	{
		GenerateListSize<T>(parameter);
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::Parameter *parameter)
	{
		if (!this->m_builder.GetInputOptions().IsVectorGeometry())
		{
			Error("tuple-in-list");
		}

		if (index == m_cellIndex)
		{
			GenerateListSize<T>(parameter);
		}
	}

private:
	template<class T>
	void GenerateListSize(const HorseIR::Parameter *parameter)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			GenerateListSize<PTX::Int8Type>(parameter);
		}
		else
		{
			auto shape = this->m_builder.GetInputOptions().ParameterShapes.at(parameter);
			auto name = NameUtils::VariableName(parameter);

			auto kernelResources = this->m_builder.GetKernelResources();
			auto kernelParameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>>(name);

			m_size = GenerateSize(kernelParameter, shape);
		}
	}

	template<class T>
	const PTX::TypedOperand<PTX::UInt32Type> *GenerateSize(const PTX::ParameterVariable<T> *parameter, const Analysis::Shape *shape) const
	{
		// Get the size register for the data

		auto resources = this->m_builder.GetLocalResources();
		auto& inputOptions = this->m_builder.GetInputOptions();

		if (const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(inputOptions.ThreadGeometry))
		{
			if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape))
			{
				return resources->GetRegister<PTX::UInt32Type>(NameUtils::SizeName(parameter));
			}
			else if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
			{
				return resources->GetRegister<PTX::UInt32Type>(NameUtils::SizeName(parameter, m_cellIndex));
			}
		}
		else if (const auto listGeometry = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(inputOptions.ThreadGeometry))
		{
			return resources->GetRegister<PTX::UInt32Type>(NameUtils::SizeName(parameter));
		}
		Error("size for shape " + Analysis::ShapeUtils::ShapeString(shape));
	}

	const PTX::TypedOperand<PTX::UInt32Type> *m_size = nullptr;

	unsigned int m_cellIndex = 0;
};

}
