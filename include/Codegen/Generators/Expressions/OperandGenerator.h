#pragma once

#include "Codegen/Generators/Generator.h"
#include "HorseIR/Traversal/ConstVisitor.h"

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

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

#include "Runtime/StringBucket.h"

#include "Utils/Logger.h"

namespace Codegen {

template<PTX::Bits B, class T>
class OperandGenerator : public Generator, public HorseIR::ConstVisitor
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
		return GenerateRegisterFromOperand(GenerateOperand(operand, index, indexName));
	}

	const PTX::Register<T> *GenerateRegister(const HorseIR::Operand *operand, LoadKind loadKind)
	{
		return GenerateRegisterFromOperand(GenerateOperand(operand, loadKind));
	}

	const PTX::Register<T> *GenerateRegisterFromOperand(const PTX::TypedOperand<T> *operand)
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
			m_operand = ConversionGenerator::ConvertSource<T, S>(this->m_builder, resources->GetRegister<S>(name));
		}
		else
		{
			// Check if the register is a parameter

			auto& parameters = this->m_builder.GetInputOptions().Parameters;
			auto& parameterShapes = this->m_builder.GetInputOptions().ParameterShapes;

			auto find = parameters.find(identifier->GetSymbol());
			if (find != parameters.end())
			{
				// Check if we have a cached register for the load kind, or need to generate the load

				auto parameter = find->second;
				if (m_index == nullptr)
				{
					auto dataIndex = GenerateIndex(parameter, m_loadKind);
					auto sourceName = NameUtils::VariableName(parameter, LoadKindString(m_loadKind));
					m_operand = ConversionGenerator::ConvertSource<T, S>(this->m_builder, GenerateParameterLoad<S>(parameter, dataIndex, sourceName));
				}
				else
				{
					auto sourceName = NameUtils::VariableName(parameter, m_indexName);
					m_operand = ConversionGenerator::ConvertSource<T, S>(this->m_builder, GenerateParameterLoad<S>(parameter, m_index, sourceName));
				}
			}
		}
		m_register = true;
	}

	void Visit(const HorseIR::BooleanLiteral *literal) override
	{
		VisitLiteral<std::int8_t>(literal);
	}

	void Visit(const HorseIR::CharLiteral *literal) override
	{
		VisitLiteral<std::int8_t>(literal);
	}

	void Visit(const HorseIR::Int8Literal *literal) override
	{
		VisitLiteral<std::int8_t>(literal);
	}

	void Visit(const HorseIR::Int16Literal *literal) override
	{
		VisitLiteral<std::int16_t>(literal);
	}

	void Visit(const HorseIR::Int32Literal *literal) override
	{
		VisitLiteral<std::int32_t>(literal);
	}

	void Visit(const HorseIR::Int64Literal *literal) override
	{
		VisitLiteral<std::int64_t>(literal);
	}

	void Visit(const HorseIR::Float32Literal *literal) override
	{
		VisitLiteral<float>(literal);
	}

	void Visit(const HorseIR::Float64Literal *literal) override
	{
		VisitLiteral<double>(literal);
	}

	void Visit(const HorseIR::StringLiteral *literal) override
	{
		VisitLiteral<std::string>(literal);
	}

	void Visit(const HorseIR::SymbolLiteral *literal) override
	{
		VisitLiteral<HorseIR::SymbolValue *>(literal);
	}

	void Visit(const HorseIR::DatetimeLiteral *literal) override
	{
		VisitLiteral<HorseIR::DatetimeValue *>(literal);
	}

	void Visit(const HorseIR::MonthLiteral *literal) override
	{
		VisitLiteral<HorseIR::MonthValue *>(literal);
	}

	void Visit(const HorseIR::DateLiteral *literal) override
	{
		VisitLiteral<HorseIR::DateValue *>(literal);
	}

	void Visit(const HorseIR::MinuteLiteral *literal) override
	{
		VisitLiteral<HorseIR::MinuteValue *>(literal);
	}

	void Visit(const HorseIR::SecondLiteral *literal) override
	{
		VisitLiteral<HorseIR::SecondValue *>(literal);
	}

	void Visit(const HorseIR::TimeLiteral *literal) override
	{
		VisitLiteral<HorseIR::TimeValue *>(literal);
	}

	template<class L>
	void VisitLiteral(const HorseIR::TypedVectorLiteral<L> *literal)
	{
		if (m_index != nullptr)
		{
			Utils::Logger::LogError("Absolute operand indexing not supported for literals");
		}

		if (literal->GetCount() == 1)
		{
			auto value = literal->GetValue(0);
			if constexpr(std::is_same<L, std::string>::value)
			{
				m_operand = new PTX::Value<T>(static_cast<typename T::SystemType>(Runtime::StringBucket::HashString(value)));
			}
			else if constexpr(std::is_same<L, HorseIR::SymbolValue *>::value)
			{
				m_operand = new PTX::Value<T>(static_cast<typename T::SystemType>(Runtime::StringBucket::HashString(value->GetName())));
			}
			else if constexpr(std::is_convertible<L, HorseIR::CalendarValue *>::value)
			{
				m_operand = new PTX::Value<T>(static_cast<typename T::SystemType>(value->GetEpochTime()));
			}
			else if constexpr(std::is_convertible<L, HorseIR::ExtendedCalendarValue *>::value)
			{
				m_operand = new PTX::Value<T>(static_cast<typename T::SystemType>(value->GetExtendedEpochTime()));
			}
			else
			{
				m_operand = new PTX::Value<T>(static_cast<typename T::SystemType>(value));
			}
		}
		else
		{
			Utils::Logger::LogError("Unsupported literal count " + std::to_string(literal->GetCount()));
		}
	}

	template<class S>
	const PTX::Register<S> *GenerateParameterLoad(const HorseIR::Parameter *parameter, const PTX::TypedOperand<PTX::UInt32Type> *dataIndex, const std::string& sourceName)
	{
		auto resources = this->m_builder.GetLocalResources();
		if (resources->ContainsRegister<S>(sourceName))
		{
			return resources->GetRegister<S>(sourceName);
		}
		else
		{
			// Ensure the thread is within bounds for loading data

			IndexGenerator indexGenerator(this->m_builder);
			auto index = indexGenerator.GenerateDataIndex();

			GeometryGenerator geometryGenerator(this->m_builder);
			auto size = geometryGenerator.GenerateDataSize();

			auto sizeLabel = this->m_builder.CreateLabel("SIZE");
			auto sizePredicate = resources->template AllocateTemporary<PTX::PredicateType>();

			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(sizePredicate, index, size, PTX::UInt32Type::ComparisonOperator::GreaterEqual));
			this->m_builder.AddStatement(new PTX::BranchInstruction(sizeLabel, sizePredicate));
			
			// Load the value from the global space

			ValueLoadGenerator<B> loadGenerator(this->m_builder);
			auto value = loadGenerator.template GenerateParameter<S>(parameter, dataIndex, sourceName);

			// Completed determining size

			this->m_builder.AddStatement(new PTX::BlankStatement());
			this->m_builder.AddStatement(sizeLabel);

			return value;
		}
	}

private:
	const PTX::Register<PTX::UInt32Type> *GenerateCompressedIndex(const Analysis::Shape::CompressedSize *size)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto& inputOptions = this->m_builder.GetInputOptions();

		auto parameter = inputOptions.ParameterObjectMap.at(size->GetPredicate());

		auto name = NameUtils::VariableName(parameter);
		auto sourceName = NameUtils::VariableName(parameter, LoadKindString(LoadKind::Vector));

		// Load the predicate parameter

		auto dataIndex = GenerateIndex(parameter, LoadKind::Vector);

		OperandGenerator<B, PTX::PredicateType> operandGenerator(this->m_builder);
		auto predicate = operandGenerator.template GenerateParameterLoad<PTX::PredicateType>(parameter, dataIndex, sourceName);

		auto intPredicate = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::SelectInstruction<PTX::UInt32Type>(intPredicate, new PTX::UInt32Value(1), new PTX::UInt32Value(0), predicate));

		// Calculate prefix sum to use as index

		auto moduleResources = this->m_builder.GetGlobalResources();
		auto g_size = moduleResources->template AllocateGlobalVariable<PTX::UInt32Type>(this->m_builder.UniqueIdentifier("size"));

		AddressGenerator<B> addressGenerator(this->m_builder);
		auto sizeAddress = addressGenerator.template GenerateAddress<PTX::UInt32Type>(g_size);

		PrefixSumGenerator<B> prefixSumGenerator(this->m_builder);
		return prefixSumGenerator.template Generate<PTX::UInt32Type>(sizeAddress, intPredicate, PrefixSumMode::Exclusive);
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateDynamicIndex(const PTX::TypedOperand<PTX::UInt32Type> *size, const PTX::TypedOperand<PTX::UInt32Type> *indexed)
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

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateIndex(const HorseIR::Parameter *parameter, const Analysis::Shape::Size *size, const Analysis::Shape::Size *geometrySize, IndexGenerator::Kind indexKind)
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
		else if (Analysis::ShapeUtils::IsCompressedSize(size, geometrySize))
		{
			// If the data is compressed, load a special prefix summed index

			return GenerateCompressedIndex(Analysis::ShapeUtils::GetSize<Analysis::Shape::CompressedSize>(size));
		}
		else
		{
			// If no static load type can be detemined, check at runtime

			auto index = indexGenerator.GenerateIndex(indexKind);

			SizeGenerator<B> sizeGenerator(this->m_builder);
			auto dynamicSize = sizeGenerator.GenerateSize(parameter);

			return GenerateDynamicIndex(dynamicSize, index);
		}
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateIndex(const HorseIR::Parameter *parameter, LoadKind loadKind)
	{
		auto& inputOptions = this->m_builder.GetInputOptions();
		auto shape = inputOptions.ParameterShapes.at(parameter);

		if (const auto vectorGeometry = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(inputOptions.ThreadGeometry))
		{
			// Vector geometries require values be loaded into vectors

			if (loadKind == LoadKind::Vector)
			{
				if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape))
				{
					return GenerateIndex(parameter, vectorShape->GetSize(), vectorGeometry->GetSize(), IndexGenerator::Kind::Global);
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

							return GenerateIndex(parameter, vectorShape->GetSize(), vectorGeometry->GetSize(), IndexGenerator::Kind::CellData);
						}
						else if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
						{
							const auto cellShape = Analysis::ShapeUtils::MergeShapes(listShape->GetElementShapes());
							if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellShape))
							{
								return GenerateIndex(parameter, vectorShape->GetSize(), vectorGeometry->GetSize(), IndexGenerator::Kind::CellData);
							}
						}
						break;
					}
					case LoadKind::ListCell:
					{
						if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape))
						{
							// As a special case, we can load vectors horizontally, with 1 value per cell

							return GenerateIndex(parameter, vectorShape->GetSize(), listGeometry->GetListSize(), IndexGenerator::Kind::Cell);
						}
						break;
					}
				}
			}
		}
		else
		{
			Utils::Logger::LogError("Unable to generate load index for thread geometry " + Analysis::ShapeUtils::ShapeString(inputOptions.ThreadGeometry));
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
