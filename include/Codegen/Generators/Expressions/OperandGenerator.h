#pragma once

#include "Codegen/Generators/Generator.h"
#include "HorseIR/Traversal/ConstVisitor.h"

#include "Analysis/Shape/Shape.h"
#include "Analysis/Shape/ShapeUtils.h"

#include "Codegen/Builder.h"
#include "Codegen/NameUtils.h"
#include "Codegen/Generators/TypeDispatch.h"
#include "Codegen/Generators/Data/ValueLoadGenerator.h"
#include "Codegen/Generators/Expressions/ConversionGenerator.h"
#include "Codegen/Generators/Expressions/MoveGenerator.h"
#include "Codegen/Generators/Indexing/AddressGenerator.h"
#include "Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Codegen/Generators/Indexing/DataSizeGenerator.h"
#include "Codegen/Generators/Indexing/PrefixSumGenerator.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "PTX/PTX.h"

#include "Runtime/StringBucket.h"

namespace Codegen {

template<PTX::Bits B, class T>
class OperandGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	using Generator::Generator;

	std::string Name() const override { return "OperandGenerator"; }

	enum class LoadKind {
		Vector,
		ListData
	};

	static std::string LoadKindString(LoadKind loadKind)
	{
		switch (loadKind)
		{
			case LoadKind::Vector:
				return "vector";
			case LoadKind::ListData:
				return "list";
		}
		return "unknown";
	}

	const PTX::TypedOperand<T> *GenerateOperand(const HorseIR::Operand *operand, const PTX::TypedOperand<PTX::UInt32Type> *index, const std::string& indexName, unsigned int cellIndex = 0)
	{
		m_index = index;
		m_indexName = indexName;
		m_cellIndex = cellIndex;

		m_operand = nullptr;
		operand->Accept(*this);
		if (m_operand != nullptr)
		{
			return m_operand;
		}

		Error("indexed operand '" + HorseIR::PrettyPrinter::PrettyString(operand) + "'");
	}

	const PTX::TypedOperand<T> *GenerateOperand(const HorseIR::Operand *operand, LoadKind loadKind, unsigned int cellIndex = 0)
	{
		m_loadKind = loadKind;

		m_index = nullptr;
		m_indexName = "";
		m_cellIndex = cellIndex;

		m_operand = nullptr;
		operand->Accept(*this);
		if (m_operand != nullptr)
		{
			return m_operand;
		}

		Error(LoadKindString(m_loadKind) + " operand '" + HorseIR::PrettyPrinter::PrettyString(operand) + "'");
	}

	const PTX::Register<T> *GenerateRegister(const HorseIR::Operand *operand, const PTX::TypedOperand<PTX::UInt32Type> *index, const std::string& indexName, unsigned int cellIndex = 0)
	{
		return GenerateRegisterFromOperand(GenerateOperand(operand, index, indexName, cellIndex));
	}

	const PTX::Register<T> *GenerateRegister(const HorseIR::Operand *operand, LoadKind loadKind, unsigned int cellIndex = 0)
	{
		return GenerateRegisterFromOperand(GenerateOperand(operand, loadKind, cellIndex));
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
	void GenerateVector(const HorseIR::Identifier *identifier)
	{
		GenerateLoad<S>(identifier, false);
	}

	template<class S>
	void GenerateList(const HorseIR::Identifier *identifier)
	{
		GenerateLoad<S>(identifier, this->m_builder.GetInputOptions().IsVectorGeometry());
	}

	template<class S>
	void GenerateTuple(unsigned int index, const HorseIR::Identifier *identifier)
	{
		if (!this->m_builder.GetInputOptions().IsVectorGeometry())
		{
			Error("tuple-in-list");
		}

		// Only generate code for the requested cell index

		if (index == m_cellIndex)
		{
			GenerateLoad<S>(identifier, true);
		}
	}

	std::string GenerateName(const HorseIR::Identifier *identifier, bool isCell) const
	{
		if (isCell)
		{
			return NameUtils::VariableName(identifier, m_cellIndex, m_indexName);
		}
		return NameUtils::VariableName(identifier, m_indexName);
	}

	std::string GenerateDestinationName(const HorseIR::Identifier *identifier, bool isCell) const
	{
		if (isCell)
		{
			return NameUtils::VariableName(identifier, m_cellIndex, (m_index == nullptr) ? LoadKindString(m_loadKind) : m_indexName);
		}
		return NameUtils::VariableName(identifier, (m_index == nullptr) ? LoadKindString(m_loadKind) : m_indexName);
	}

	template<class S>
	void GenerateLoad(const HorseIR::Identifier *identifier, bool isCell)
	{
		// Determine if the identifier is a local variable or parameter

		auto resources = this->m_builder.GetLocalResources();

		// Check if the register has been assigned (or re-assigned for parameters)

		auto name = GenerateName(identifier, isCell);
		auto destinationName = GenerateDestinationName(identifier, isCell);

		const PTX::Register<S> *operandRegister = nullptr;
		if (resources->ContainsRegister<S>(name))
		{
			operandRegister = resources->GetRegister<S>(name);
		}
		else
		{
			// Check if the register is a parameter

			auto& parameters = this->m_builder.GetInputOptions().Parameters;

			auto find = parameters.find(identifier->GetSymbol());
			if (find != parameters.end())
			{
				// Check if we have a cached register for the load kind, or need to generate the load

				auto parameter = find->second;
				operandRegister = GenerateParameterLoad<S>(destinationName, parameter, nullptr, isCell);
			}
		}

		m_operand = ConversionGenerator::ConvertSource<T, S>(this->m_builder, operandRegister);
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
			Error("indexed literal operand");
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
			Error("literal count " + std::to_string(literal->GetCount()));
		}
	}

	template<class S>
	const PTX::Register<S> *GenerateParameterLoad(const std::string& name, const HorseIR::Parameter *parameter, const PTX::TypedOperand<PTX::UInt32Type> *dataIndex, bool isCell = false)
	{
		auto resources = this->m_builder.GetLocalResources();
		if (resources->ContainsRegister<S>(name))
		{
			return resources->GetRegister<S>(name);
		}
		else
		{
			if constexpr(std::is_same<S, PTX::PredicateType>::value)
			{
				auto value = GenerateParameterLoad<PTX::Int8Type>(name, parameter, dataIndex, isCell);
				return ConversionGenerator::ConvertSource<S, PTX::Int8Type>(this->m_builder, value);
			}
			else
			{
				auto kernelResources = this->m_builder.GetKernelResources();

				// Ensure the thread is within bounds for loading data

				if (dataIndex == nullptr)
				{
					dataIndex = (m_index == nullptr) ? GenerateIndex(parameter, m_loadKind) : m_index;
				}

				DataSizeGenerator<B> sizeGenerator(this->m_builder);
				auto size = (isCell) ? sizeGenerator.GenerateSize(parameter, m_cellIndex) : sizeGenerator.GenerateSize(parameter);

				auto sizeLabel = this->m_builder.CreateLabel("SIZE");
				auto sizePredicate = resources->template AllocateTemporary<PTX::PredicateType>();

				this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(sizePredicate, dataIndex, size, PTX::UInt32Type::ComparisonOperator::GreaterEqual));
				this->m_builder.AddStatement(new PTX::BranchInstruction(sizeLabel, sizePredicate));

				// Load the value from the global space
				
				ValueLoadGenerator<B, S> loadGenerator(this->m_builder);
				const PTX::Register<S> *value = nullptr;

				auto shape = this->m_builder.GetInputOptions().ParameterShapes.at(parameter);
				if (Analysis::ShapeUtils::IsShape<Analysis::VectorShape>(shape))
				{
					auto kernelParameter = kernelResources->template GetParameter<PTX::PointerType<B, S>>(NameUtils::VariableName(parameter));
					value = loadGenerator.GeneratePointer(name, kernelParameter, dataIndex);
				}
				else if (Analysis::ShapeUtils::IsShape<Analysis::ListShape>(shape))
				{
					// The parameter is unindexed, the cell is added in the value generation

					auto kernelParameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::PointerType<B, S, PTX::GlobalSpace>>>(NameUtils::VariableName(parameter));
					if (isCell)
					{
						value = loadGenerator.GeneratePointer(name, kernelParameter, m_cellIndex, dataIndex);
					}
					else
					{
						value = loadGenerator.GeneratePointer(name, kernelParameter, dataIndex);
					}
				}


				// Completed determining size

				this->m_builder.AddStatement(new PTX::BlankStatement());
				this->m_builder.AddStatement(sizeLabel);

				if (m_compressionRegister != nullptr)
				{
					resources->template SetCompressedRegister<S>(value, m_compressionRegister);
				}

				return value;
			}
		}
	}

private:
	const PTX::Register<PTX::UInt32Type> *GenerateCompressedIndex(const Analysis::Shape::CompressedSize *size)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto& inputOptions = this->m_builder.GetInputOptions();

		auto parameter = inputOptions.ParameterObjectMap.at(size->GetPredicate());
		auto name = NameUtils::VariableName(parameter, LoadKindString(LoadKind::Vector));

		// Load the predicate parameter

		auto dataIndex = GenerateIndex(parameter, LoadKind::Vector);

		OperandGenerator<B, PTX::PredicateType> operandGenerator(this->m_builder);
		m_compressionRegister = operandGenerator.template GenerateParameterLoad<PTX::PredicateType>(name, parameter, dataIndex);

		auto prefixCompression = resources->template GetCompressedRegister<PTX::PredicateType>(m_compressionRegister);

		// Calculate prefix sum to use as index

		PrefixSumGenerator<B, PTX::UInt32Type> prefixSumGenerator(this->m_builder);
		return prefixSumGenerator.template Generate<PTX::PredicateType>(m_compressionRegister, PrefixSumMode::Exclusive, prefixCompression);
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateDynamicIndex(const PTX::TypedOperand<PTX::UInt32Type> *size, const PTX::TypedOperand<PTX::UInt32Type> *indexed)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Generate a dynamic check for the size of a parameter. If the value is a scalar, we use null addressing,
		// otherwise we use the indexing mode passed to this function

		auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
		auto index = resources->template AllocateTemporary<PTX::UInt32Type>();

		DataIndexGenerator<B> indexGenerator(this->m_builder);
		auto null = indexGenerator.GenerateIndex(DataIndexGenerator<B>::Kind::Broadcast);

		// Select the right index using the size

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predicate, size, new PTX::UInt32Value(1), PTX::UInt32Type::ComparisonOperator::Equal));
		this->m_builder.AddStatement(new PTX::SelectInstruction<PTX::UInt32Type>(index, null, indexed, predicate));

		return index;
	}

	const PTX::TypedOperand<PTX::UInt32Type> *GenerateIndex(const HorseIR::Parameter *parameter, const Analysis::Shape::Size *size, const Analysis::Shape::Size *geometrySize, typename DataIndexGenerator<B>::Kind indexKind, bool isCell = false)
	{
		DataIndexGenerator<B> indexGenerator(this->m_builder);

		if (const auto constantSize = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(size))
		{
			// If the vector is a constant scalar, broadcast

			if (constantSize->GetValue() == 1)
			{
				return indexGenerator.GenerateIndex(DataIndexGenerator<B>::Kind::Broadcast);
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

			DataSizeGenerator<B> sizeGenerator(this->m_builder);
			const PTX::TypedOperand<PTX::UInt32Type> *dynamicSize = nullptr;

			if (isCell)
			{
				dynamicSize = sizeGenerator.GenerateSize(parameter, m_cellIndex);
			}
			else
			{
				dynamicSize = sizeGenerator.GenerateSize(parameter);
			}

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
					return GenerateIndex(parameter, vectorShape->GetSize(), vectorGeometry->GetSize(), DataIndexGenerator<B>::Kind::VectorData);
				}
				else if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
				{
					const auto cellShape = Analysis::ShapeUtils::MergeShapes(listShape->GetElementShapes());
					if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellShape))
					{
						// Load data using the vector indexing scheme

						return GenerateIndex(parameter, vectorShape->GetSize(), vectorGeometry->GetSize(), DataIndexGenerator<B>::Kind::VectorData, true);
					}
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

							return GenerateIndex(parameter, vectorShape->GetSize(), vectorGeometry->GetSize(), DataIndexGenerator<B>::Kind::ListData);
						}
						else if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
						{
							const auto cellShape = Analysis::ShapeUtils::MergeShapes(listShape->GetElementShapes());
							if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(cellShape))
							{
								return GenerateIndex(parameter, vectorShape->GetSize(), vectorGeometry->GetSize(), DataIndexGenerator<B>::Kind::ListData);
							}
						}
						break;
					}
					case LoadKind::ListData:
					{
						if (const auto vectorShape = Analysis::ShapeUtils::GetShape<Analysis::VectorShape>(shape))
						{
							// As a special case, we can load vectors horizontally, with 1 value per cell

							return GenerateIndex(parameter, vectorShape->GetSize(), listGeometry->GetListSize(), DataIndexGenerator<B>::Kind::ListBroadcast);
						}
						break;
					}
				}
			}
		}
		Error(LoadKindString(loadKind) + " load index for shape " + Analysis::ShapeUtils::ShapeString(shape));
	}

	const PTX::TypedOperand<T> *m_operand = nullptr;
	const PTX::Register<PTX::PredicateType> *m_compressionRegister = nullptr;
	bool m_register = false;

	LoadKind m_loadKind;

	const PTX::TypedOperand<PTX::UInt32Type> *m_index = nullptr;
	std::string m_indexName = "";
	unsigned int m_cellIndex = 0;
};

}
