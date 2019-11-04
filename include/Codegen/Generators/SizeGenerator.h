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

#include "Utils/Logger.h"

namespace Codegen {

template<PTX::Bits B>
class SizeGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	using Generator::Generator;

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateSize(const HorseIR::Parameter *parameter)
	{
		m_size = nullptr;
		DispatchType(*this, parameter->GetType(), parameter);
		if (m_size == nullptr)
		{
			Utils::Logger::LogError("Unable to determine size for parameter " + HorseIR::PrettyPrinter::PrettyString(parameter));
		}
		return m_size;
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateSize(const HorseIR::Operand *operand)
	{
		m_size = nullptr;
		operand->Accept(*this);
		if (m_size == nullptr)
		{
			Utils::Logger::LogError("Unable to determine size for operand " + HorseIR::PrettyPrinter::PrettyString(operand));
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
	void Generate(const HorseIR::Parameter *parameter)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			Generate<PTX::Int8Type>(parameter);
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
				m_size = GenerateSize(parameter, shape, inputOptions.ThreadGeometry);
			}
			else if (Analysis::ShapeUtils::IsShape<Analysis::ListShape>(shape))
			{
				auto parameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>>(name);
				m_size = GenerateSize(parameter, shape, inputOptions.ThreadGeometry);
			}
		}
	}

	template<class T>
	const PTX::TypedOperand<PTX::UInt32Type> *GenerateSize(const PTX::ParameterVariable<T> *parameter, const Analysis::Shape *shape, const Analysis::Shape *threadGeometry) const
	{
		if (const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(threadGeometry))
		{
			if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape))
			{
				return GenerateSize(parameter, vectorShape->GetSize(), vectorGeometry->GetSize());
			}
		}
		else if (const auto listGeometry = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(threadGeometry))
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
		return nullptr;
	}

	template<class T>
	const PTX::TypedOperand<PTX::UInt32Type> *GenerateSize(const PTX::ParameterVariable<T> *parameter, const Analysis::Shape::Size *size, const Analysis::Shape::Size *geometrySize) const
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
			return resources->GetRegister<PTX::UInt32Type>(NameUtils::GeometryDataSize);
		}
		else
		{
			// Size is not statically determined, load at runtime from special register

			return resources->GetRegister<PTX::UInt32Type>(NameUtils::SizeName(parameter));
		}
	}

private:
	const PTX::TypedOperand<PTX::UInt32Type> *m_size = nullptr;
};

}
