#pragma once

#include "Codegen/Generators/Generator.h"
#include "HorseIR/Traversal/ConstVisitor.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Data/ValueLoadGenerator.h"
#include "Codegen/Generators/Expressions/LiteralGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/PTX.h"

#include "Utils/Logger.h"

namespace Codegen {

template<PTX::Bits B>
class OrderLoadGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	OrderLoadGenerator(Builder& builder, const PTX::Register<PTX::UInt32Type> *leftIndex, const PTX::Register<PTX::UInt32Type> *rightIndex) :
		Generator(builder), m_leftIndex(leftIndex), m_rightIndex(rightIndex) {}

	void Generate(const std::vector<HorseIR::Operand *>& dataArguments)
	{
		for (const auto argument : dataArguments)
		{
			argument->Accept(*this);
		}
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	void Visit(const HorseIR::Literal *literal) override
	{
		Utils::Logger::LogError("Unsupported literal operand for order function");
	}

	template<class T>
	void Generate(const HorseIR::Identifier *identifier)
	{
		OperandGenerator<B, T> operandGenerator(this->m_builder);
		operandGenerator.GenerateOperand(identifier, m_leftIndex, "left");
		operandGenerator.GenerateOperand(identifier, m_rightIndex, "right");
	}

private:
	const PTX::Register<PTX::UInt32Type> *m_leftIndex = nullptr;
	const PTX::Register<PTX::UInt32Type> *m_rightIndex = nullptr;
};

template<PTX::Bits B>
class OrderComparisonGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	enum class Order {
		Ascending,
		Descending
	};

	OrderComparisonGenerator(Builder& builder, Order sequenceOrder, const PTX::Label *swapLabel, const PTX::Label *endLabel) :
		Generator(builder), m_sequenceOrder(sequenceOrder), m_swapLabel(swapLabel), m_endLabel(endLabel) {}

	void Generate(const std::vector<HorseIR::Operand *>& dataArguments, const HorseIR::TypedVectorLiteral<char> *orderLiteral)
	{
		auto index = 0;
		for (const auto argument : dataArguments)
		{
			m_dataOrder = (orderLiteral->GetValue(index)) ? Order::Ascending : Order::Descending;
			m_hasNext = (index + 1 < dataArguments.size());
			argument->Accept(*this);
			index++;
		}
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	void Visit(const HorseIR::Literal *literal) override
	{
		Utils::Logger::LogError("Unsupported literal operand for order function");
	}

	template<class T>
	void Generate(const HorseIR::Identifier *identifier)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value || std::is_same<T, PTX::Int8Type>::value)
		{
			Generate<PTX::Int16Type>(identifier);
		}
		else
		{
			OperandGenerator<B, T> operandGenerator(this->m_builder);
			auto leftValue = operandGenerator.GenerateOperand(identifier, nullptr, "left");
			auto rightValue = operandGenerator.GenerateOperand(identifier, nullptr, "right");
			Generate<T>(leftValue, rightValue);
		}
	}

	template<class T>
	void Generate(const PTX::TypedOperand<T> *leftValue, const PTX::TypedOperand<T> *rightValue)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto predicateSwap = resources->template AllocateTemporary<PTX::PredicateType>();

		if (m_sequenceOrder == m_dataOrder)
		{
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(predicateSwap, leftValue, rightValue, T::ComparisonOperator::Less));
		}
		else
		{
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(predicateSwap, leftValue, rightValue, T::ComparisonOperator::Greater));
		}

		// Branch if the predicate is true

		this->m_builder.AddStatement(new PTX::BranchInstruction(m_swapLabel, predicateSwap));
		
		if (m_hasNext)
		{
			// Check for the next branch

			auto predicateEqual = resources->template AllocateTemporary<PTX::PredicateType>();

			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(predicateEqual, leftValue, rightValue, T::ComparisonOperator::NotEqual));
			this->m_builder.AddStatement(new PTX::BranchInstruction(m_endLabel, predicateEqual));
		}
	}

private:
	const PTX::Label *m_swapLabel = nullptr;
	const PTX::Label *m_endLabel = nullptr;

	bool m_hasNext = false;
	Order m_dataOrder;
	Order m_sequenceOrder;
};

template<PTX::Bits B>
class OrderSwapGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	OrderSwapGenerator(Builder& builder, const PTX::Register<PTX::UInt32Type> *leftIndex, const PTX::Register<PTX::UInt32Type> *rightIndex) :
		Generator(builder), m_leftIndex(leftIndex), m_rightIndex(rightIndex) {}

	void Generate(const HorseIR::Operand *index, const std::vector<HorseIR::Operand *>& dataArguments)
	{
		index->Accept(*this);
		for (const auto argument : dataArguments)
		{
			argument->Accept(*this);
		}
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	void Visit(const HorseIR::Literal *literal) override
	{
		Utils::Logger::LogError("Unsupported literal operand for order function");
	}

	template<class T>
	void Generate(const HorseIR::Identifier *identifier)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			Generate<PTX::Int8Type>(identifier);
		}
		else
		{
			// Swap the left and right values in global memory

			auto resources = this->m_builder.GetLocalResources();
			auto kernelResources = this->m_builder.GetKernelResources();
			
			// Get the left and right values

			OperandGenerator<B, T> operandGenerator(this->m_builder);
			auto leftValue = operandGenerator.GenerateRegister(identifier, m_leftIndex, "left");
			auto rightValue = operandGenerator.GenerateRegister(identifier, m_rightIndex, "right");

			// Get the addresses of the left and right positions

			auto parameter = kernelResources->template GetParameter<PTX::PointerType<B, T>>(NameUtils::VariableName(identifier));

			AddressGenerator<B> addressGenerator(this->m_builder);
			auto leftAddress = addressGenerator.template GenerateAddress<T>(parameter, m_leftIndex);
			auto rightAddress = addressGenerator.template GenerateAddress<T>(parameter, m_rightIndex);

			// Store the results back

			this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(leftAddress, rightValue));
			this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(rightAddress, leftValue));
		}
	}

private:
	const PTX::Register<PTX::UInt32Type> *m_leftIndex = nullptr;
	const PTX::Register<PTX::UInt32Type> *m_rightIndex = nullptr;
};

template<PTX::Bits B>
class OrderGenerator : public Generator
{
public:
	using Generator::Generator;

	void Generate(const std::vector<HorseIR::Operand *>& arguments)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Add the special parameters for sorting (stage and substage) and load the values

		ParameterGenerator<B> parameterGenerator(this->m_builder);
		auto sortStageParameter = parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::SortStage);
		auto sortSubstageParameter = parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::SortSubstage);

		ValueLoadGenerator<B> valueLoadGenerator(this->m_builder);
		auto stage = valueLoadGenerator.template GenerateConstant<PTX::UInt32Type>(sortStageParameter);
		auto substage = valueLoadGenerator.template GenerateConstant<PTX::UInt32Type>(sortSubstageParameter);

		IndexGenerator indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateGlobalIndex();

		// Compute the size of each bitonic sequence in this stage
		//   sequenceSize = 2^(stage + 1)

		auto temp1 = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto temp2 = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto sequenceSize = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(temp1, stage, new PTX::UInt32Value(1)));
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(temp2, new PTX::UInt32Value(1)));
		this->m_builder.AddStatement(new PTX::ShiftLeftInstruction<PTX::Bit32Type>(
			new PTX::Bit32RegisterAdapter<PTX::UIntType>(sequenceSize),
			new PTX::Bit32Adapter<PTX::UIntType>(temp2),
			temp1
		));

		// Compute the sequence index of this thread. We allocate threads for half the sequence size as each thread will perform 1 swap
		//   sequenceIndex = (index / (sequenceSize / 2))

		auto temp_halfSequence = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto sequenceIndex = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::ShiftRightInstruction<PTX::Bit32Type>(
			new PTX::Bit32RegisterAdapter<PTX::UIntType>(temp_halfSequence),
			new PTX::Bit32Adapter<PTX::UIntType>(sequenceSize),
			new PTX::UInt32Value(1)
		));
		this->m_builder.AddStatement(new PTX::DivideInstruction<PTX::UInt32Type>(sequenceIndex, index, temp_halfSequence));

		// Compute the sequence start index for this thread
		//   sequenceStart = sequenceindex * sequenceSize

		auto sequenceStart = resources->template AllocateTemporary<PTX::UInt32Type>();

		auto multiplyInstruction = new PTX::MultiplyInstruction<PTX::UInt32Type>(sequenceStart, sequenceIndex, sequenceSize);
		multiplyInstruction->SetLower(true);
		this->m_builder.AddStatement(multiplyInstruction);

		// === Find the substage location

		// Compute the size of each substage
		//   subsequenceSize = sequenceSize >> substage

		auto subsequenceSize = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::ShiftRightInstruction<PTX::Bit32Type>(
			new PTX::Bit32RegisterAdapter<PTX::UIntType>(subsequenceSize),
			new PTX::Bit32Adapter<PTX::UIntType>(sequenceSize),
			substage
		));

		// Compute the index of the substage, again half the number of threads are active
		//   subsequenceIndex = (index % (sequenceSize / 2)) / (subsequenceSize / 2);

		auto temp4 = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto temp_halfSubsequence = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto subsequenceIndex = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(temp4, index, temp_halfSequence));
		this->m_builder.AddStatement(new PTX::ShiftRightInstruction<PTX::Bit32Type>(
			new PTX::Bit32RegisterAdapter<PTX::UIntType>(temp_halfSubsequence),
			new PTX::Bit32Adapter<PTX::UIntType>(subsequenceSize),
			new PTX::UInt32Value(1)
		));
		this->m_builder.AddStatement(new PTX::DivideInstruction<PTX::UInt32Type>(subsequenceIndex, temp4, temp_halfSubsequence));

		// Compute the subsequence start index for this thread
		//   subsequenceStart = sequenceStart + (subsequenceIndex * subsequenceSize)

		auto subsequenceStart = resources->template AllocateTemporary<PTX::UInt32Type>();

		auto madInstruction = new PTX::MADInstruction<PTX::UInt32Type>(subsequenceStart, subsequenceIndex, subsequenceSize, sequenceStart);
		madInstruction->SetLower(true);
		this->m_builder.AddStatement(madInstruction);

		// Index of the thread in *its* substage
		//   subsequenceLocalIndex = index % (subsequenceSize / 2)

		auto subsequenceLocalIndex = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::RemainderInstruction<PTX::UInt32Type>(subsequenceLocalIndex, index, temp_halfSubsequence));

		// Compute the indices of the left and right data items
		//   leftIndex = subsequenceStart + subsequenceLocalIndex
		//   rightIndex = leftIndex + (subsequenceSize / 2)

		auto leftIndex = resources->template AllocateTemporary<PTX::UInt32Type>();
		auto rightIndex = resources->template AllocateTemporary<PTX::UInt32Type>();

		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(leftIndex, subsequenceStart, subsequenceLocalIndex));
		this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(rightIndex, leftIndex, temp_halfSubsequence));

		// Convenience for separating the data from the index

		std::vector<HorseIR::Operand *> dataArguments(std::begin(arguments) + 1, std::end(arguments) - 1);
		const auto& indexArgument = arguments.at(0);
		const auto& orderArgument = arguments.at(arguments.size() - 1);
		auto orderLiteral = LiteralGenerator<char>::GetLiteral(orderArgument);

		// Load the left and right values

		OrderLoadGenerator<B> loadGenerator(this->m_builder, leftIndex, rightIndex);
		loadGenerator.Generate(dataArguments);

		// Generate the if-else structure for the sort order

		auto temp5 = resources->template AllocateTemporary<PTX::Bit32Type>();
		auto subsequenceDirection = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::AndInstruction<PTX::Bit32Type>(
			temp5, new PTX::Bit32Adapter<PTX::UIntType>(sequenceIndex), new PTX::Bit32Adapter<PTX::UIntType>(new PTX::UInt32Value(1))
		));
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::Bit32Type>(
			subsequenceDirection, temp5, new PTX::Bit32Adapter<PTX::UIntType>(new PTX::UInt32Value(1)), PTX::Bit32Type::ComparisonOperator::NotEqual
		));

		auto elseLabel = this->m_builder.CreateLabel("ELSE");
		auto swapLabel = this->m_builder.CreateLabel("SWAP");
		auto endLabel = this->m_builder.CreateLabel("END");

		this->m_builder.AddStatement(new PTX::BranchInstruction(elseLabel, subsequenceDirection));
		this->m_builder.AddStatement(new PTX::BlankStatement());

		// True branch (ascending sequence)

		OrderComparisonGenerator<B> ascendingGenerator(this->m_builder, OrderComparisonGenerator<B>::Order::Ascending, swapLabel, endLabel);
		ascendingGenerator.Generate(dataArguments, orderLiteral);

		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel));

		// Else branch (descending sequence)

		this->m_builder.AddStatement(elseLabel);
		this->m_builder.AddStatement(new PTX::BlankStatement());

		OrderComparisonGenerator<B> descendingGenerator(this->m_builder, OrderComparisonGenerator<B>::Order::Descending, swapLabel, endLabel);
		descendingGenerator.Generate(dataArguments, orderLiteral);

		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel));

		// Swap if needed!

		this->m_builder.AddStatement(swapLabel);

		OrderSwapGenerator<B> swapGenerator(this->m_builder, leftIndex, rightIndex);
		swapGenerator.Generate(indexArgument, dataArguments);

		// Finally, we end the order

		this->m_builder.AddStatement(endLabel);
	}
};

}
