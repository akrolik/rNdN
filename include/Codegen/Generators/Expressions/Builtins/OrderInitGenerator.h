#pragma once

#include <limits>

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/SizeGenerator.h"
#include "Codegen/Generators/TargetGenerator.h"
#include "Codegen/Generators/Expressions/ConversionGenerator.h"
#include "Codegen/Generators/Expressions/LiteralGenerator.h"
#include "Codegen/Generators/Expressions/MoveGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class OrderInitValueGenerator : public Generator
{
public:
	using Generator::Generator;

	template<class T>
	void Generate(const HorseIR::LValue *target, const HorseIR::Operand *operand)
	{
		// Load the sort data into registers

		OperandGenerator<B, T> opGen(this->m_builder);
		auto value = opGen.GenerateOperand(operand, OperandGenerator<B, T>::LoadKind::Vector);

		// Get the target register

		TargetGenerator<B, T> targetGenerator(this->m_builder);
		auto targetRegister = targetGenerator.Generate(target, nullptr);
		
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

	enum class Order {
		Ascending,
		Descending
	};

	template<class T>
	void Generate(const HorseIR::LValue *target, const HorseIR::Operand *operand, Order order)
	{
		// Compute the null value depending on the sort order

		auto null = ((order == Order::Ascending) ? std::numeric_limits<typename T::SystemType>::max() : std::numeric_limits<typename T::SystemType>::min());
		auto value = new PTX::Value<T>(null);

		// Get the target register

		TargetGenerator<B, T> targetGenerator(this->m_builder);
		auto targetRegister = targetGenerator.Generate(target, nullptr);
		
		// Move the null value into the target

		MoveGenerator<T> moveGenerator(this->m_builder);
		moveGenerator.Generate(targetRegister, value);
	}
};

template<PTX::Bits B>
class OrderInitGenerator : public Generator
{
public:
	using Generator::Generator;

	void Generate(const std::vector<HorseIR::LValue *>& targets, const std::vector<HorseIR::Type *>& returnTypes, const std::vector<HorseIR::Operand *>& arguments)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Generate the index value for the thread and convert to the right type

		IndexGenerator indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateGlobalIndex();

		TargetGenerator<B, PTX::Int64Type> targetGenerator(this->m_builder);
		auto indexRegister = targetGenerator.Generate(targets.at(0), nullptr);

		ConversionGenerator::ConvertSource(this->m_builder, indexRegister, index);

		// Compute the number of active data items - the rest will be nulled. We assume all data columns are the same size

		SizeGenerator<B> sizeGenerator(this->m_builder);
		auto size = sizeGenerator.GenerateSize(arguments.at(0));

		// Generate the if-else structure

		auto elseLabel = this->m_builder.CreateLabel("ELSE");
		auto endLabel = this->m_builder.CreateLabel("END");
		auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(predicate, index, size, PTX::UInt32Type::ComparisonOperator::GreaterEqual));
		this->m_builder.AddStatement(new PTX::BranchInstruction(elseLabel, predicate));
		this->m_builder.AddStatement(new PTX::BlankStatement());

		// Decompose the arguments into data and order

		std::vector<HorseIR::Operand *> dataArguments(std::begin(arguments), std::end(arguments) - 1);
		const auto& orderArgument = arguments.at(arguments.size() - 1);

		auto orderLiteral = LiteralGenerator<char>::GetLiteral(orderArgument);

		// True branch (index < size), load the values into the target registers, skipping the index target

		auto targetIndex = 1;
		OrderInitValueGenerator<B> valueGenerator(this->m_builder);
		for (const auto operand : dataArguments)
		{
			DispatchType(valueGenerator, operand->GetType(), targets.at(targetIndex), operand);
			targetIndex++;
		}

		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel));

		// Else branch (index >= size), load the min/max values depending on sort

		this->m_builder.AddStatement(elseLabel);
		this->m_builder.AddStatement(new PTX::BlankStatement());

		targetIndex = 1;
		OrderInitNullGenerator<B> nullGenerator(this->m_builder);
		for (const auto operand : dataArguments)
		{
			auto order = (orderLiteral->GetValue(targetIndex - 1)) ? OrderInitNullGenerator<B>::Order::Ascending : OrderInitNullGenerator<B>::Order::Descending;
			DispatchType(nullGenerator, operand->GetType(), targets.at(targetIndex), operand, order);
			targetIndex++;
		}

		this->m_builder.AddStatement(endLabel);
	}
};

}
