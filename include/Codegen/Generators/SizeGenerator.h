#pragma once

#include "Codegen/Generators/Generator.h"
#include "HorseIR/Traversal/ConstVisitor.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"

#include "PTX/PTX.h"

namespace Codegen {

class SizeGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	using Generator::Generator;

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
		auto& inputOptions = this->m_builder.GetInputOptions();
		auto& parameterShapes = inputOptions.ParameterShapes;

		auto symbol = identifier->GetSymbol();
		if (parameterShapes.find(symbol) != parameterShapes.end())
		{
			auto shape = parameterShapes.at(symbol);   
			m_size = GenerateSize(identifier, shape, inputOptions.ThreadGeometry);
		}
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateSize(const HorseIR::Identifier *identifier, const Analysis::Shape *shape, const Analysis::Shape *threadGeometry) const
	{
		if (const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(threadGeometry))
		{
			if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape))
			{
				return GenerateSize(identifier, vectorShape->GetSize(), vectorGeometry->GetSize());
			}
		}
		else if (const auto listGeometry = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(threadGeometry))
		{
			const auto cellGeometry = Analysis::ShapeUtils::MergeShapes(listGeometry->GetElementShapes());
			if (const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellGeometry))
			{
				if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape))
				{
					return GenerateSize(identifier, vectorShape->GetSize(), vectorGeometry->GetSize());
				}
				else if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
				{
					const auto cellShape = Analysis::ShapeUtils::MergeShapes(listShape->GetElementShapes());
					if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellShape))
					{
						return GenerateSize(identifier, vectorShape->GetSize(), vectorGeometry->GetSize());
					}
				}
			}
		}
		return nullptr;
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateSize(const HorseIR::Identifier *identifier, const Analysis::Shape::Size *size, const Analysis::Shape::Size *geometrySize) const
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
			// Size is not statically determined, load at runtime

			auto name = NameUtils::SizeName(NameUtils::VariableName(identifier));
			return resources->GetRegister<PTX::UInt32Type>(name);
		}
	}

private:
	const PTX::TypedOperand<PTX::UInt32Type> *m_size = nullptr;
};

}
