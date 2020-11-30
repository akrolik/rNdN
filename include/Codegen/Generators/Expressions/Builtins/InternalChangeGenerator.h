#pragma once

#include "Codegen/Generators/Generator.h"
#include "HorseIR/Traversal/ConstVisitor.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/Builtins/ComparisonGenerator.h"
#include "Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Codegen/Generators/Indexing/ThreadGeometryGenerator.h"

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Codegen {

template<PTX::Bits B>
class InternalChangeGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	using Generator::Generator;

	std::string Name() const override { return "InternalChangeGenerator"; }

	const PTX::Register<PTX::PredicateType> *Generate(const HorseIR::Operand *dataArgument)
	{
		// Initialize the current and previous values, and compute the change
		//   
		//   1. Check if geometry is within bounds
		//   2. Load the current value
		//   3. Load the previous value at index -1 (bounded below by index 0)

		m_predicate = nullptr;
		dataArgument->Accept(*this);

		// Check the geometry is within bounds, if so we will load both values and do a comparison

		ThreadGeometryGenerator<B> geometryGenerator(this->m_builder);
		auto geometry = geometryGenerator.GenerateDataGeometry();

		DataIndexGenerator<B> indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateDataIndex();

		auto resources = this->m_builder.GetLocalResources();
		auto boundPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		auto diffPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		auto firstPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
		auto changePredicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(boundPredicate, index, geometry, PTX::UInt32Type::ComparisonOperator::Less));
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(firstPredicate, index, new PTX::UInt32Value(0), PTX::UInt32Type::ComparisonOperator::Equal));
		this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(diffPredicate, boundPredicate, m_predicate));
		this->m_builder.AddStatement(new PTX::OrInstruction<PTX::PredicateType>(changePredicate, diffPredicate, firstPredicate));

		return changePredicate;
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	template<class T>
	void GenerateVector(const HorseIR::Identifier *identifier)
	{
		GenerateChange<T>(identifier);
	}

	template<class T>
	void GenerateList(const HorseIR::Identifier *identifier)
	{
		const auto& inputOptions = this->m_builder.GetInputOptions();
		const auto parameter = inputOptions.Parameters.at(identifier->GetSymbol());
		const auto shape = inputOptions.ParameterShapes.at(parameter);

		if (const auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape))
		{
			if (const auto size = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(listShape->GetListSize()))
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

		GenerateChange<T>(identifier, true, index);
	}
 
	template<class T>
	void GenerateChange(const HorseIR::Identifier *identifier, bool isCell = false, unsigned int cellIndex = 0)
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

		auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();

		ComparisonGenerator<B, PTX::PredicateType> comparisonGenerator(this->m_builder, ComparisonOperation::NotEqual);
		comparisonGenerator.Generate(predicate, value, previousValue);

		// Merge previous columns, one must be different

		if (m_predicate == nullptr)
		{
			m_predicate = predicate;
		}
		else
		{
			this->m_builder.AddStatement(new PTX::OrInstruction<PTX::PredicateType>(m_predicate, m_predicate, predicate));
		}
	}
                                                                                                                  
private:
	const PTX::Register<PTX::PredicateType> *m_predicate = nullptr;
};
 
}
