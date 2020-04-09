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

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateSize(const HorseIR::Parameter *parameter, unsigned int cellIndex)
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

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateSize(const HorseIR::Parameter *parameter)
	{
		m_cellIndex = 0;
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
		auto& inputOptions = this->m_builder.GetInputOptions();
		if (const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(inputOptions.ThreadGeometry))
		{
			if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape))
			{
				return GenerateSize(parameter, vectorShape->GetSize(), vectorGeometry->GetSize());
			}
			else if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
			{
				const auto cellShape = Analysis::ShapeUtils::MergeShapes(listShape->GetElementShapes());
				if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellShape))
				{
					return GenerateSize(parameter, vectorShape->GetSize(), vectorGeometry->GetSize(), true);
				}
			}
		}
		else if (const auto listGeometry = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(inputOptions.ThreadGeometry))
		{
			const auto cellGeometry = Analysis::ShapeUtils::MergeShapes(listGeometry->GetElementShapes());
			if (const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellGeometry))
			{
				if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape))
				{
					return GenerateSize(parameter, vectorShape->GetSize(), vectorGeometry->GetSize());
				}
				else if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
				{
					const auto cellShape = Analysis::ShapeUtils::MergeShapes(listShape->GetElementShapes());
					if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellShape))
					{
						return GenerateSize(parameter, vectorShape->GetSize(), vectorGeometry->GetSize());
					}
				}
			}
		}
		Error("size for shape " + Analysis::ShapeUtils::ShapeString(shape));
	}

	template<class T>
	const PTX::TypedOperand<PTX::UInt32Type> *GenerateSize(const PTX::ParameterVariable<T> *parameter, const Analysis::Shape::Size *size, const Analysis::Shape::Size *geometrySize, bool isCell = false) const
	{
		auto resources = this->m_builder.GetLocalResources();

		if (const auto constantSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(size))
		{
			// Statically determined constant size

			return new PTX::UInt32Value(constantSize->GetValue());
		}
		else if (*size == *geometrySize)
		{
			// If the size is equal to the geometry, then we return the size of the geometry

			if (const auto constantSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(geometrySize))
			{
				return new PTX::UInt32Value(constantSize->GetValue());
			}
			return resources->GetRegister<PTX::UInt32Type>(NameUtils::ThreadGeometryDataSize);
		}

		// Size is not statically determined, load at runtime from special register

		if (isCell)
		{
			return resources->GetRegister<PTX::UInt32Type>(NameUtils::SizeName(parameter, m_cellIndex));
		}
		return resources->GetRegister<PTX::UInt32Type>(NameUtils::SizeName(parameter));
	}

	const PTX::TypedOperand<PTX::UInt32Type> *m_size = nullptr;

	unsigned int m_cellIndex = 0;
};

}
