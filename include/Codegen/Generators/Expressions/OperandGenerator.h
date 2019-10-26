#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"
#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"
#include "Codegen/Generators/IndexGenerator.h"
#include "Codegen/Generators/SizeGenerator.h"
#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/Data/ValueLoadGenerator.h"
#include "Codegen/Generators/Expressions/ConversionGenerator.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "PTX/PTX.h"

#include "Utils/Logger.h"

namespace Codegen {

template<PTX::Bits B, class T>
class OperandGenerator : public HorseIR::ConstVisitor, public Generator
{
public:
	using Generator::Generator;

	enum class LoadKind {
		Vector,
		ListCell
	};

	static std::string LoadKindString(LoadKind loadKind)
	{
		switch (loadKind)
		{
			case LoadKind::Vector:
				return "vector";
			case LoadKind::ListCell:
				return "cell";
		}
		return "unknown";
	}

	const PTX::TypedOperand<T> *GenerateOperand(const HorseIR::Expression *expression, LoadKind loadKind)
	{
		m_loadKind = loadKind;

		m_operand = nullptr;
		expression->Accept(*this);
		if (m_operand != nullptr)
		{
			return m_operand;
		}

		Utils::Logger::LogError("Unable to generate operand '" + HorseIR::PrettyPrinter::PrettyString(expression) + "'");
	}

	const PTX::Register<T> *GenerateRegister(const HorseIR::Expression *expression, LoadKind loadKind)
	{
		const PTX::TypedOperand<T> *operand = GenerateOperand(expression, loadKind);
		if (m_register)
		{
			return static_cast<const PTX::Register<T> *>(operand);
		}

		auto resources = this->m_builder.GetLocalResources();

		if constexpr(std::is_same<T, PTX::Int8Type>::value)
		{
			auto destination = resources->template AllocateTemporary<PTX::Int8Type>();
			auto bracedSource = new PTX::Braced2Operand<PTX::Bit8Type>({
				new PTX::Bit8Adapter<PTX::IntType>(operand),
				new PTX::Value<PTX::Bit8Type>(0)
			});
			auto bracedTarget = new PTX::Braced2Register<PTX::Bit8Type>({
				new PTX::Bit8RegisterAdapter<PTX::IntType>(destination),
				new PTX::SinkRegister<PTX::Bit8Type>
			});
			auto temp = resources->template AllocateTemporary<PTX::Bit16Type>();

			this->m_builder.AddStatement(new PTX::Pack2Instruction<PTX::Bit16Type>(temp, bracedSource));
			this->m_builder.AddStatement(new PTX::Unpack2Instruction<PTX::Bit16Type>(bracedTarget, temp));

			return destination;
		}
		else
		{
			auto destination = resources->template AllocateTemporary<T>();
			this->m_builder.AddStatement(new PTX::MoveInstruction<T>(destination, operand));
			return destination;
		}
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	template<class S>
	void Generate(const HorseIR::Identifier *identifier)
	{
		// Determine if the identifier is a local variable or parameter

		auto resources = this->m_builder.GetLocalResources();
		auto name = NameUtils::VariableName(identifier->GetName());

		// Check if the register has been assigned (or re-assigned for parameters)

		if (resources->ContainsRegister<S>(name))
		{
			GenerateRegister(resources->GetRegister<S>(name));
		}
		else
		{
			// Check if the register is a parameter

			auto& parameterShapes = this->m_builder.GetInputOptions().ParameterShapes;
			if (parameterShapes.find(identifier->GetSymbol()) != parameterShapes.end())
			{
				// Check if we have a cached register for the load kind, or need to generate the load

				auto sourceName = NameUtils::VariableName(identifier->GetName(), LoadKindString(m_loadKind));
				if (resources->ContainsRegister<S>(sourceName))
				{
					GenerateRegister(resources->GetRegister<S>(sourceName));
				}
				else
				{
					// Generate the load according to the load kind and data size

					auto shape = parameterShapes.at(identifier->GetSymbol());   
					auto index = GetIndex(identifier->GetName(), shape, m_loadKind);

					ValueLoadGenerator<B> loadGenerator(this->m_builder);
					GenerateRegister(loadGenerator.template GeneratePointer<S>(name, index, sourceName));
				}
			}
		}
	}

	template<class S>
	void GenerateRegister(const PTX::Register<S> *source)
	{
		//GLOBAL: Identifiers may be global and have a module name
		if constexpr(std::is_same<T, S>::value)
		{
			m_operand = source;
		}
		else
		{
			m_operand = ConversionGenerator::ConvertSource<T, S>(this->m_builder, source);
		}
		m_register = true;
	}

	void Visit(const HorseIR::BooleanLiteral *literal) override
	{
		Generate<char>(literal);
	}

	void Visit(const HorseIR::CharLiteral *literal) override
	{
		Generate<char>(literal);
	}

	void Visit(const HorseIR::Int8Literal *literal) override
	{
		Generate<std::int8_t>(literal);
	}

	void Visit(const HorseIR::Int16Literal *literal) override
	{
		Generate<std::int16_t>(literal);
	}

	void Visit(const HorseIR::Int32Literal *literal) override
	{
		Generate<std::int32_t>(literal);
	}

	void Visit(const HorseIR::Int64Literal *literal) override
	{
		Generate<std::int64_t>(literal);
	}

	void Visit(const HorseIR::Float32Literal *literal) override
	{
		Generate<float>(literal);
	}

	void Visit(const HorseIR::Float64Literal *literal) override
	{
		Generate<double>(literal);
	}

	void Visit(const HorseIR::DateLiteral *literal) override
	{
		//TODO: Extend to other date types
		if (literal->GetCount() == 1)
		{
			m_operand = new PTX::Value<T>(literal->GetValue(0)->GetEpochTime());
		}
		else
		{
			Utils::Logger::LogError("Unsupported literal count " + std::to_string(literal->GetCount()));
		}
	}

	template<class L>
	void Generate(const HorseIR::TypedVectorLiteral<L> *literal)
	{
		if (literal->GetCount() == 1)
		{
			if constexpr(std::is_same<typename T::SystemType, L>::value)
			{
				m_operand = new PTX::Value<T>(literal->GetValue(0));
			}
			else
			{
				m_operand = new PTX::Value<T>(static_cast<typename T::SystemType>(literal->GetValue(0)));
			}
		}
		else
		{
			Utils::Logger::LogError("Unsupported literal count " + std::to_string(literal->GetCount()));
		}
	}

private:
	const PTX::TypedOperand<PTX::UInt32Type> *GetSizeIndex(const PTX::TypedOperand<PTX::UInt32Type> *size, const PTX::TypedOperand<PTX::UInt32Type> *indexed)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Generate a dynamic check for the size of a parameter. If the value is a scalar, we use null addressing,
		// otherwise we use the indexing mode passed to this function

		auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
		auto index = resources->template AllocateTemporary<PTX::UInt32Type>();

		IndexGenerator indexGenerator(this->m_builder);
		auto null = indexGenerator.GenerateIndex(IndexGenerator::Kind::Null);

		// Select the right index using the size

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predicate, size, new PTX::UInt32Value(1), PTX::UInt32Type::ComparisonOperator::Equal));
		this->m_builder.AddStatement(new PTX::SelectInstruction<PTX::UInt32Type>(index, null, indexed, predicate));

		return index;
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GetIndex(const std::string& name, const Analysis::Shape::Size *size, const Analysis::Shape::Size *geometrySize, IndexGenerator::Kind indexKind, bool list)
	{
		IndexGenerator indexGenerator(this->m_builder);

		if (const auto constantSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(size))
		{
			// If the vector is a constant scalar, broadcast

			if (constantSize->GetValue() == 1)
			{
				return indexGenerator.GenerateIndex(IndexGenerator::Kind::Null);
			}

			//Otherwise, use the supplied indexing mode

			return indexGenerator.GenerateIndex(indexKind);
		}
		else if (*size == *geometrySize)
		{
			// If the geometry is equal, we assume this is due to a vector write

			return indexGenerator.GenerateIndex(indexKind);                      
		}
		else
		{
			// If no static load type can be detemined, check at runtime

			SizeGenerator sizeGenerator(this->m_builder);
			auto index = indexGenerator.GenerateIndex(indexKind);
			if (list)
			{
				auto size = sizeGenerator.GenerateVectorSize(name);
				return GetSizeIndex(size, index);
			}
			else
			{
				auto size = sizeGenerator.GenerateCellSize(name);
				return GetSizeIndex(size, index);
			}
		}
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GetIndex(const std::string& name, const Analysis::Shape *shape, LoadKind loadKind)
	{
		auto& inputOptions = this->m_builder.GetInputOptions();
		if (const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(inputOptions.ThreadGeometry))
		{
			// Vector geometries require values be loaded into vectors

			if (loadKind == LoadKind::Vector)
			{
				if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape))
				{
					return GetIndex(name, vectorShape->GetSize(), vectorGeometry->GetSize(), IndexGenerator::Kind::Global, false);
				}
			}
		}
		else if (const auto listGeometry = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(inputOptions.ThreadGeometry))
		{
			// Ensure the list geometry contents is a collection of vectors

			const auto cellGeometry = Analysis::ShapeUtils::MergeShapes(listGeometry->GetElementShapes());
			if (const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellGeometry))
			{
				switch (loadKind)
				{
					case LoadKind::Vector:
					{
						if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape))
						{
							// Vectors loaded in list geometries align with the cell data (vertical vectors)

							return GetIndex(name, vectorShape->GetSize(), vectorGeometry->GetSize(), IndexGenerator::Kind::CellData, false);
						}
						else if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
						{
							const auto cellShape = Analysis::ShapeUtils::MergeShapes(listShape->GetElementShapes());
							if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellShape))
							{
								return GetIndex(name, vectorShape->GetSize(), vectorGeometry->GetSize(), IndexGenerator::Kind::CellData, true);
							}
						}
						break;
					}
					case LoadKind::ListCell:
					{
						if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape))
						{
							// As a special case, we can load vectors horizontally, with 1 value per cell

							return GetIndex(name, vectorShape->GetSize(), listGeometry->GetListSize(), IndexGenerator::Kind::Cell, false);
						}
						break;
					}
				}
			}
		}
		Utils::Logger::LogError("Unable to determine indexing mode for shape " + Analysis::ShapeUtils::ShapeString(shape) + " and load kind " + LoadKindString(loadKind));
	}

	const PTX::TypedOperand<T> *m_operand = nullptr;
	bool m_register = false;

	LoadKind m_loadKind;
};

}
