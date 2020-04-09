#pragma once

#include "Codegen/Generators/Generator.h"
#include "HorseIR/Traversal/ConstVisitor.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Indexing/ThreadGeometryGenerator.h"
#include "Codegen/Generators/Indexing/DataIndexGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class GroupChangeGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	using Generator::Generator;

	std::string Name() const override { return "GroupChangeGenerator"; }

	const PTX::Register<PTX::PredicateType> *Generate(const HorseIR::Operand *dataArgument)
	{
		m_predicate = nullptr;
		dataArgument->Accept(*this);
		return m_predicate;
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	template<class T>
	void GenerateVector(const HorseIR::Identifier *identifier)
	{
		GenerateGroup<T>(identifier);
	}

	template<class T>
	void GenerateList(const HorseIR::Identifier *identifier)
	{
		const auto& inputOptions = this->m_builder.GetInputOptions();
		const auto parameter = inputOptions.Parameters.at(identifier->GetSymbol());
		const auto shape = inputOptions.ParameterShapes.at(parameter);

		if (const auto listShape = Analysis::ShapeUtils::GetShape<Analysis::ListShape>(shape))
		{
			if (const auto size = Analysis::ShapeUtils::GetSize<Analysis::Shape::ConstantSize>(listShape->GetListSize()))
			{
				for (auto index = 0u; index < size->GetValue(); ++index)
				{
					GenerateTuple<T>(index, identifier);
				}
				return;
			}
		}
		Error("non-constant cell count");
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::Identifier *identifier)
	{
		if (!this->m_builder.GetInputOptions().IsVectorGeometry())
		{
			Error("tuple-in-list");
		}

		GenerateGroup<T>(identifier, true, index);
	}
 
	template<class T>
	void GenerateGroup(const HorseIR::Identifier *identifier, bool isCell = false, unsigned int cellIndex = 0)
	{
		//TODO: Use comparison generator instead to generate optimized comparison
		if constexpr(std::is_same<T, PTX::PredicateType>::value || std::is_same<T, PTX::Int8Type>::value)
		{
			GenerateGroup<PTX::Int16Type>(identifier, isCell, cellIndex);
		}
		else
		{
			auto resources = this->m_builder.GetLocalResources();

			// Load the current value index

			DataIndexGenerator<B> indexGenerator(this->m_builder);
			auto index = indexGenerator.GenerateDataIndex();

			// Load the previous value index , duplicating the first element (as there is no previous element)

			auto indexM1 = resources->template AllocateTemporary<PTX::UInt32Type>();
			this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(indexM1, index, new PTX::UInt32Value(1)));

			auto previousPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
				previousPredicate, index, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::NotEqual
			)); 

			auto indexPrevious = resources->template AllocateTemporary<PTX::UInt32Type>();
			this->m_builder.AddStatement(new PTX::SelectInstruction<PTX::UInt32Type>(indexPrevious, indexM1, index, previousPredicate));

			// Get the operand values

			OperandGenerator<B, T> opGen(this->m_builder);
			const PTX::TypedOperand<T> *value = nullptr;
			const PTX::TypedOperand<T> *previousValue = nullptr;

			if (isCell)
			{
				value = opGen.GenerateOperand(identifier, index, "val", cellIndex);
				previousValue = opGen.GenerateOperand(identifier, indexPrevious, "prev", cellIndex);
			}
			else
			{
				value = opGen.GenerateOperand(identifier, index, "val");
				previousValue = opGen.GenerateOperand(identifier, indexPrevious, "prev");
			}

			// Check if the value is different

			if (m_predicate == nullptr)
			{
				m_predicate = resources->template AllocateTemporary<PTX::PredicateType>();
				this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(m_predicate, value, previousValue, T::ComparisonOperator::NotEqual)); 
			}
			else
			{
				this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(
					m_predicate, nullptr, value, previousValue, T::ComparisonOperator::NotEqual, m_predicate, PTX::PredicateModifier::BoolOperator::Or
				)); 
			}
		}
	}
                                                                                                                  
private:
	const PTX::Register<PTX::PredicateType> *m_predicate = nullptr;
};

template<PTX::Bits B>
class GroupGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "GroupGenerator"; }

	void Generate(const std::vector<HorseIR::LValue *>& targets, const std::vector<HorseIR::Operand *>& arguments)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Convenience split of input arguments

		const auto indexArgument = arguments.at(0);
		const auto dataArgument = arguments.at(1);

		// Initialize the current and previous values, and compute the change
		//   
		//   1. Check if geometry is within bounds
		//   2. Load the current value
		//   3. Load the previous value at index -1 (bounded below by index 0)

		// Check for each column if there has been a change

		GroupChangeGenerator<B> changeGenerator(this->m_builder);
		auto changePredicate = changeGenerator.Generate(dataArgument);

		// Check the geometry is within bounds, if so we will load both values and do a comparison

		ThreadGeometryGenerator<B> geometryGenerator(this->m_builder);
		auto geometry = geometryGenerator.GenerateDataGeometry();

		DataIndexGenerator<B> indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateDataIndex();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
			changePredicate, nullptr, index, geometry, PTX::UInt32Type::ComparisonOperator::Less, changePredicate, PTX::PredicateModifier::BoolOperator::And
		));

		// Get the return targets!

		TargetGenerator<B, PTX::Int64Type> targetGenerator(this->m_builder);
		auto keys = targetGenerator.Generate(targets.at(0), nullptr);
		auto values = targetGenerator.Generate(targets.at(1), nullptr);

		// Set the key as the dataIndex value

		auto compressPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
			compressPredicate, nullptr, index, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::Equal, changePredicate, PTX::PredicateModifier::BoolOperator::Or
		));

		OperandGenerator<B, PTX::Int64Type> operandGenerator(this->m_builder);
		auto dataIndex = operandGenerator.GenerateOperand(indexArgument, index, this->m_builder.UniqueIdentifier("index"));

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(keys, dataIndex));

		resources->SetCompressedRegister(keys, compressPredicate);

		// Set the value as the index into the dataIndex

		ConversionGenerator::ConvertSource(this->m_builder, values, index);
		resources->SetCompressedRegister(values, compressPredicate);
	}
};

}
