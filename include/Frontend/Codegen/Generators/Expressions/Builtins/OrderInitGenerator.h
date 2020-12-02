#pragma once

#include <limits>

#include "Frontend/Codegen/Generators/Generator.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Data/TargetGenerator.h"
#include "Frontend/Codegen/Generators/Data/TargetCellGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/ConversionGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/MoveGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/OperandGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataSizeGenerator.h"

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/LiteralUtils.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B>
class OrderInitValueGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "OrderInitValueGenerator"; }

	void Generate(const HorseIR::LValue *target, const HorseIR::Operand *dataArgument)
	{
		DispatchType(*this, dataArgument->GetType(), target, dataArgument);
	}

	template<class T>
	void GenerateVector(const HorseIR::LValue *target, const HorseIR::Operand *dataArgument)
	{
		// Load the sort data into registers

		OperandGenerator<B, T> operandGenerator(this->m_builder);
		operandGenerator.SetBoundsCheck(false);
		auto value = operandGenerator.GenerateOperand(dataArgument, OperandGenerator<B, T>::LoadKind::Vector);

		// Get the target register

		TargetGenerator<B, T> targetGenerator(this->m_builder);
		auto targetRegister = targetGenerator.Generate(target, nullptr);
		
		// Move the actual value into the target

		MoveGenerator<T> moveGenerator(this->m_builder);
		moveGenerator.Generate(targetRegister, value);
	}

	template<class T>
	void GenerateList(const HorseIR::LValue *target, const HorseIR::Operand *dataArgument)
	{
		const auto& inputOptions = this->m_builder.GetInputOptions();
		const auto declaration = inputOptions.Declarations.at(target->GetSymbol());
		const auto shape = inputOptions.DeclarationShapes.at(declaration);

		if (const auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape))
		{
			if (const auto size = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(listShape->GetListSize()))
			{
				for (auto index = 0u; index < size->GetValue(); ++index)
				{
					GenerateTuple<T>(index, target, dataArgument);
				}
				return;
			}
		}
		Error("non-constant cell count");
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::LValue *target, const HorseIR::Operand *dataArgument)
	{
		// Load the sort data into registers

		OperandGenerator<B, T> operandGenerator(this->m_builder);
		operandGenerator.SetBoundsCheck(false);
		auto value = operandGenerator.GenerateOperand(dataArgument, OperandGenerator<B, T>::LoadKind::Vector, index);

		// Get the target register

		TargetCellGenerator<B, T> targetGenerator(this->m_builder);
		auto targetRegister = targetGenerator.Generate(target, index);
		
		// Move the actual value into the target

		MoveGenerator<T> moveGenerator(this->m_builder);
		moveGenerator.Generate(targetRegister, value);
	}
};

template<PTX::Bits B>
class OrderInitNullGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "OrderInitNullGenerator"; }

	enum class Order {
		Ascending,
		Descending
	};

	void Generate(const HorseIR::LValue *target, const HorseIR::Operand *dataArgument, const HorseIR::TypedVectorLiteral<std::int8_t> *orderLiteral)
	{
		DispatchType(*this, dataArgument->GetType(), target, dataArgument, orderLiteral);
	}

	template<class T>
	void GenerateVector(const HorseIR::LValue *target, const HorseIR::Operand *dataArgument, const HorseIR::TypedVectorLiteral<std::int8_t> *orderLiteral)
	{
		// Compute the null value depending on the sort order

		auto order = GenerateOrder(orderLiteral->GetValue(0));
		auto value = GenerateNull<T>(order);

		// Get the target register

		TargetGenerator<B, T> targetGenerator(this->m_builder);
		auto targetRegister = targetGenerator.Generate(target, nullptr);
		
		// Move the null value into the target

		MoveGenerator<T> moveGenerator(this->m_builder);
		moveGenerator.Generate(targetRegister, value);
	}

	template<class T>
	void GenerateList(const HorseIR::LValue *target, const HorseIR::Operand *dataArgument, const HorseIR::TypedVectorLiteral<std::int8_t> *orderLiteral)
	{
		const auto& inputOptions = this->m_builder.GetInputOptions();
		const auto declaration = inputOptions.Declarations.at(target->GetSymbol());
		const auto shape = inputOptions.DeclarationShapes.at(declaration);

		if (const auto listShape = HorseIR::Analysis::ShapeUtils::GetShape<HorseIR::Analysis::ListShape>(shape))
		{
			if (const auto size = HorseIR::Analysis::ShapeUtils::GetSize<HorseIR::Analysis::Shape::ConstantSize>(listShape->GetListSize()))
			{
				for (auto index = 0u; index < size->GetValue(); ++index)
				{
					GenerateTuple<T>(index, target, dataArgument, orderLiteral);
				}
				return;
			}
		}
		Error("non-constant cell count");
	}

	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::LValue *target, const HorseIR::Operand *dataArgument, const HorseIR::TypedVectorLiteral<std::int8_t> *orderLiteral)
	{
		auto order = GenerateOrder(orderLiteral->GetValue((orderLiteral->GetCount() == 1 ? 0 : index)));
		auto value = GenerateNull<T>(order);

		// Get the target register

		TargetCellGenerator<B, T> targetGenerator(this->m_builder);
		auto targetRegister = targetGenerator.Generate(target, index);
		
		// Move the null value into the target

		MoveGenerator<T> moveGenerator(this->m_builder);
		moveGenerator.Generate(targetRegister, value);
	}

private:
	Order GenerateOrder(bool order) const
	{
		if (order)
		{
			return OrderInitNullGenerator<B>::Order::Ascending;
		}
		return OrderInitNullGenerator<B>::Order::Descending;      
	}

	template<class T>
	const PTX::Value<T> *GenerateNull(Order order) const
	{
		if (order == Order::Ascending)
		{
			return new PTX::Value<T>(std::numeric_limits<typename T::SystemType>::max());
		}
		return new PTX::Value<T>(std::numeric_limits<typename T::SystemType>::min());
	}
};

template<PTX::Bits B>
class OrderInitGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "OrderInitGenerator"; }

	void Generate(const std::vector<HorseIR::LValue *>& targets, const std::vector<HorseIR::Operand *>& arguments)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Generate the index value for the thread and convert to the right type

		DataIndexGenerator<B> indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateDataIndex();

		TargetGenerator<B, PTX::Int64Type> targetGenerator(this->m_builder);
		auto indexRegister = targetGenerator.Generate(targets.at(0), nullptr);

		ConversionGenerator::ConvertSource(this->m_builder, indexRegister, index);

		// Compute the number of active data items - the rest will be nulled. We assume all data columns are the same size

		DataSizeGenerator<B> sizeGenerator(this->m_builder);
		auto size = sizeGenerator.GenerateSize(arguments.at(0));

		// Generate the if-else structure

		auto elseLabel = this->m_builder.CreateLabel("ELSE");
		auto endLabel = this->m_builder.CreateLabel("END");
		auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predicate, index, size, PTX::UInt32Type::ComparisonOperator::GreaterEqual));
		this->m_builder.AddStatement(new PTX::BranchInstruction(elseLabel, predicate));
		this->m_builder.AddStatement(new PTX::BlankStatement());

		// Decompose the arguments into data and order

		const auto& dataArgument = arguments.at(0);
		const auto& orderArgument = arguments.at(1);

		auto orderLiteral = HorseIR::LiteralUtils<std::int8_t>::GetLiteral(orderArgument);

		// True branch (index < size), load the values into the target registers, skipping the index target

		OrderInitValueGenerator<B> valueGenerator(this->m_builder);
		valueGenerator.Generate(targets.at(1), dataArgument);

		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel));

		// Else branch (index >= size), load the min/max values depending on sort

		this->m_builder.AddStatement(elseLabel);
		this->m_builder.AddStatement(new PTX::BlankStatement());

		OrderInitNullGenerator<B> nullGenerator(this->m_builder);
		nullGenerator.Generate(targets.at(1), dataArgument, orderLiteral);

		this->m_builder.AddStatement(endLabel);
	}
};

}
}
