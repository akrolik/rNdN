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
#include "Codegen/Generators/Expressions/MoveGenerator.h"

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

	const PTX::TypedOperand<T> *GenerateOperand(const HorseIR::Operand *operand, const PTX::TypedOperand<PTX::UInt32Type> *index, const std::string& indexName = "")
	{
		m_index = index;
		m_indexName = indexName;

		m_operand = nullptr;
		operand->Accept(*this);
		if (m_operand != nullptr)
		{
			return m_operand;
		}

		Utils::Logger::LogError("Unable to generate indexed operand '" + HorseIR::PrettyPrinter::PrettyString(operand) + "'");
	}

	const PTX::TypedOperand<T> *GenerateOperand(const HorseIR::Operand *operand, LoadKind loadKind)
	{
		m_loadKind = loadKind;
		m_index = nullptr;
		m_indexName = "";

		m_operand = nullptr;
		operand->Accept(*this);
		if (m_operand != nullptr)
		{
			return m_operand;
		}

		Utils::Logger::LogError("Unable to generate operand '" + HorseIR::PrettyPrinter::PrettyString(operand) + "'");
	}

	const PTX::Register<T> *GenerateRegister(const HorseIR::Operand *operand, const PTX::TypedOperand<PTX::UInt32Type> *index, const std::string& indexName = "")
	{
		return GenerateRegister(GenerateOperand(operand, index, indexName));
	}

	const PTX::Register<T> *GenerateRegister(const HorseIR::Operand *operand, LoadKind loadKind)
	{
		return GenerateRegister(GenerateOperand(operand, loadKind));
	}

	const PTX::Register<T> *GenerateRegister(const PTX::TypedOperand<T> *operand)
	{
		if (m_register)
		{
			return static_cast<const PTX::Register<T> *>(operand);
		}

		// Move the value into a register

		auto resources = this->m_builder.GetLocalResources();
		auto destination = resources->template AllocateTemporary<T>();

		MoveGenerator<T> moveGenerator(this->m_builder);
		moveGenerator.Generate(destination, operand);

		return destination;
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
		auto name = NameUtils::VariableName(identifier, m_indexName);

		// Check if the register has been assigned (or re-assigned for parameters)

		if (resources->ContainsRegister<S>(name))
		{
			GenerateRegister(resources->GetRegister<S>(name));
		}
		else
		{
			// Check if the register is a parameter

			auto& parameterShapes = this->m_builder.GetInputOptions().ParameterShapes;

			auto find = parameterShapes.find(identifier->GetSymbol());
			if (find != parameterShapes.end())
			{
				// Check if we have a cached register for the load kind, or need to generate the load

				auto sourceName = NameUtils::VariableName(identifier, LoadKindString(m_loadKind));
				if (resources->ContainsRegister<S>(sourceName))
				{
					GenerateRegister(resources->GetRegister<S>(sourceName));
				}
				else
				{
					// Generate the load according to the load kind and data size or absolutely using the supplied index

					GenerateParameterLoad<S>(identifier, find->second);
				}
			}
		}
	}

	template<class S>
	void GenerateParameterLoad(const HorseIR::Identifier *identifier, const Analysis::Shape *shape)
	{
		if constexpr(std::is_same<S, PTX::PredicateType>::value)
		{
			// Boolean parameters are stored as 8-bit integers

			GenerateParameterLoad<PTX::Int8Type>(identifier, shape);
		}
		else
		{
			auto kernelResources = this->m_builder.GetKernelResources();

			auto name = NameUtils::VariableName(identifier);
			auto sourceName = NameUtils::VariableName(identifier, m_indexName);

			auto index = (m_index == nullptr) ? GetIndex(identifier, shape, m_loadKind) : m_index;

			ValueLoadGenerator<B> loadGenerator(this->m_builder);
			if (Analysis::ShapeUtils::IsShape<Analysis::VectorShape>(shape))
			{
				auto parameter = kernelResources->template GetParameter<PTX::PointerType<B, S>>(name);
				GenerateRegister(loadGenerator.template GeneratePointer<S>(parameter, index, sourceName));
			}
			else
			{
				auto parameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::PointerType<B, S, PTX::GlobalSpace>>>(name);
				GenerateRegister(loadGenerator.template GeneratePointer<S>(parameter, index, sourceName));
			}
		}
	}

	template<class S>
	void GenerateRegister(const PTX::Register<S> *source)
	{
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
		if (m_index != nullptr)
		{
			Utils::Logger::LogError("Absolute operand indexing not supported for literals");
		}

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
		if (m_index != nullptr)
		{
			Utils::Logger::LogError("Absolute operand indexing not supported for literals");
		}

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

	const PTX::TypedOperand<PTX::UInt32Type> *GetIndex(const HorseIR::Identifier *identifier, const Analysis::Shape::Size *size, const Analysis::Shape::Size *geometrySize, IndexGenerator::Kind indexKind)
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

			auto index = indexGenerator.GenerateIndex(indexKind);

			SizeGenerator<B> sizeGenerator(this->m_builder);
			auto dynamicSize = sizeGenerator.GenerateSize(identifier);

			return GetSizeIndex(dynamicSize, index);
		}
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GetIndex(const HorseIR::Identifier *identifier, const Analysis::Shape *shape, LoadKind loadKind)
	{
		auto& inputOptions = this->m_builder.GetInputOptions();
		if (const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(inputOptions.ThreadGeometry))
		{
			// Vector geometries require values be loaded into vectors

			if (loadKind == LoadKind::Vector)
			{
				if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape))
				{
					return GetIndex(identifier, vectorShape->GetSize(), vectorGeometry->GetSize(), IndexGenerator::Kind::Global);
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

							return GetIndex(identifier, vectorShape->GetSize(), vectorGeometry->GetSize(), IndexGenerator::Kind::CellData);
						}
						else if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
						{
							const auto cellShape = Analysis::ShapeUtils::MergeShapes(listShape->GetElementShapes());
							if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellShape))
							{
								return GetIndex(identifier, vectorShape->GetSize(), vectorGeometry->GetSize(), IndexGenerator::Kind::CellData);
							}
						}
						break;
					}
					case LoadKind::ListCell:
					{
						if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape))
						{
							// As a special case, we can load vectors horizontally, with 1 value per cell

							return GetIndex(identifier, vectorShape->GetSize(), listGeometry->GetListSize(), IndexGenerator::Kind::Cell);
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

	const PTX::TypedOperand<PTX::UInt32Type> *m_index = nullptr;
	std::string m_indexName = "";
};

}
