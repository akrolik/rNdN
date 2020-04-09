#pragma once

#include "Codegen/Generators/Generator.h"
#include "HorseIR/Traversal/ConstVisitor.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Data/ParameterGenerator.h"
#include "Codegen/Generators/Data/ValueLoadGenerator.h"
#include "Codegen/Generators/Expressions/OperandGenerator.h"
#include "Codegen/Generators/Indexing/AddressGenerator.h"
#include "Codegen/Generators/Indexing/DataIndexGenerator.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/LiteralUtils.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class OrderLoadGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	OrderLoadGenerator(Builder& builder, const PTX::Register<PTX::UInt32Type> *leftIndex, const PTX::Register<PTX::UInt32Type> *rightIndex) :
		Generator(builder), m_leftIndex(leftIndex), m_rightIndex(rightIndex) {}

	std::string Name() const override { return "OrderLoadGenerator"; }

	void Generate(const HorseIR::Operand *dataArgument)
	{
		dataArgument->Accept(*this);
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	void Visit(const HorseIR::Literal *literal) override
	{
		Error("literal data for @order");
	}

	template<class T>
	void GenerateVector(const HorseIR::Identifier *identifier)
	{
		OperandGenerator<B, T> operandGenerator(this->m_builder);
		operandGenerator.GenerateOperand(identifier, m_leftIndex, "left");
		operandGenerator.GenerateOperand(identifier, m_rightIndex, "right");
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
		OperandGenerator<B, T> operandGenerator(this->m_builder);
		operandGenerator.GenerateOperand(identifier, m_leftIndex, "left", index);
		operandGenerator.GenerateOperand(identifier, m_rightIndex, "right", index);
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

	std::string Name() const override { return "OrderComparisonGenerator"; }

	void Generate(const HorseIR::Operand *dataArgument, const HorseIR::TypedVectorLiteral<std::int8_t> *orderLiteral)
	{
		m_orderLiteral = orderLiteral;
		dataArgument->Accept(*this);
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	void Visit(const HorseIR::Literal *literal) override
	{
		Error("literal data for @order");
	}

	template<class T>
	void GenerateVector(const HorseIR::Identifier *identifier)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value || std::is_same<T, PTX::Int8Type>::value)
		{
			GenerateVector<PTX::Int16Type>(identifier);
		}
		else
		{
			OperandGenerator<B, T> operandGenerator(this->m_builder);
			auto leftValue = operandGenerator.GenerateOperand(identifier, nullptr, "left");
			auto rightValue = operandGenerator.GenerateOperand(identifier, nullptr, "right");
			Generate<T>(leftValue, rightValue, 0);
		}
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
		if constexpr(std::is_same<T, PTX::PredicateType>::value || std::is_same<T, PTX::Int8Type>::value)
		{
			GenerateTuple<PTX::Int16Type>(index, identifier);
		}
		else
		{
			OperandGenerator<B, T> operandGenerator(this->m_builder);
			auto leftValue = operandGenerator.GenerateOperand(identifier, nullptr, "left", index);
			auto rightValue = operandGenerator.GenerateOperand(identifier, nullptr, "right", index);
			Generate<T>(leftValue, rightValue, index);
		}
	}

	template<class T>
	void Generate(const PTX::TypedOperand<T> *leftValue, const PTX::TypedOperand<T> *rightValue, unsigned int index)
	{
		auto resources = this->m_builder.GetLocalResources();
		auto predicateSwap = resources->template AllocateTemporary<PTX::PredicateType>();

		auto dataOrder = (m_orderLiteral->GetValue((m_orderLiteral->GetCount() == 1 ? 0 : index))) ? Order::Ascending : Order::Descending;
		if (m_sequenceOrder == dataOrder)
		{
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(predicateSwap, leftValue, rightValue, T::ComparisonOperator::Less));
		}
		else
		{
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(predicateSwap, leftValue, rightValue, T::ComparisonOperator::Greater));
		}

		// Branch if the predicate is true

		this->m_builder.AddStatement(new PTX::BranchInstruction(m_swapLabel, predicateSwap));
		
		if (index + 1 < m_orderLiteral->GetCount())
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

	const HorseIR::TypedVectorLiteral<std::int8_t> *m_orderLiteral = nullptr;
	Order m_sequenceOrder;
};

template<PTX::Bits B>
class OrderSwapGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	OrderSwapGenerator(Builder& builder, const PTX::Register<PTX::UInt32Type> *leftIndex, const PTX::Register<PTX::UInt32Type> *rightIndex) :
		Generator(builder), m_leftIndex(leftIndex), m_rightIndex(rightIndex) {}

	std::string Name() const override { return "OrderSwapGenerator"; }

	void Generate(const HorseIR::Operand *index, const HorseIR::Operand *dataArgument)
	{
		index->Accept(*this);
		dataArgument->Accept(*this);
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	void Visit(const HorseIR::Literal *literal) override
	{
		Error("literal data for @order");
	}

	template<class T>
	void GenerateVector(const HorseIR::Identifier *identifier)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			GenerateVector<PTX::Int8Type>(identifier);
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

			AddressGenerator<B, T> addressGenerator(this->m_builder);
			auto leftAddress = addressGenerator.GenerateAddress(parameter, m_leftIndex);
			auto rightAddress = addressGenerator.GenerateAddress(parameter, m_rightIndex);

			// Store the results back

			this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(leftAddress, rightValue));
			this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(rightAddress, leftValue));
		}
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
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			GenerateTuple<PTX::Int8Type>(index, identifier);
		}
		else
		{
			// Swap the left and right values in global memory

			auto resources = this->m_builder.GetLocalResources();
			auto kernelResources = this->m_builder.GetKernelResources();
			
			// Get the left and right values

			OperandGenerator<B, T> operandGenerator(this->m_builder);
			auto leftValue = operandGenerator.GenerateRegister(identifier, m_leftIndex, "left", index);
			auto rightValue = operandGenerator.GenerateRegister(identifier, m_rightIndex, "right", index);

			// Get the addresses of the left and right positions

			auto parameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>>(NameUtils::VariableName(identifier));

			AddressGenerator<B, T> addressGenerator(this->m_builder);
			auto leftAddress = addressGenerator.GenerateAddress(parameter, index, m_leftIndex);
			auto rightAddress = addressGenerator.GenerateAddress(parameter, index, m_rightIndex);

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

	std::string Name() const override { return "OrderGenerator"; }

	void Generate(const std::vector<HorseIR::Operand *>& arguments)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Add the special parameters for sorting (stage and substage) and load the values

		ParameterGenerator<B> parameterGenerator(this->m_builder);
		auto sortStageParameter = parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::SortStage);
		auto sortSubstageParameter = parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::SortSubstage);

		ValueLoadGenerator<B, PTX::UInt32Type> valueLoadGenerator(this->m_builder);
		auto stage = valueLoadGenerator.GenerateConstant(sortStageParameter);
		auto substage = valueLoadGenerator.GenerateConstant(sortSubstageParameter);

		DataIndexGenerator<B> indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateDataIndex();

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

		const auto& indexArgument = arguments.at(0);
		const auto& dataArgument = arguments.at(1);
		const auto& orderArgument = arguments.at(2);
		auto orderLiteral = HorseIR::LiteralUtils<std::int8_t>::GetLiteral(orderArgument);

		// Load the left and right values

		OrderLoadGenerator<B> loadGenerator(this->m_builder, leftIndex, rightIndex);
		loadGenerator.Generate(dataArgument);

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
		ascendingGenerator.Generate(dataArgument, orderLiteral);

		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel));

		// Else branch (descending sequence)

		this->m_builder.AddStatement(elseLabel);
		this->m_builder.AddStatement(new PTX::BlankStatement());

		OrderComparisonGenerator<B> descendingGenerator(this->m_builder, OrderComparisonGenerator<B>::Order::Descending, swapLabel, endLabel);
		descendingGenerator.Generate(dataArgument, orderLiteral);

		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel));

		// Swap if needed!

		this->m_builder.AddStatement(swapLabel);

		OrderSwapGenerator<B> swapGenerator(this->m_builder, leftIndex, rightIndex);
		swapGenerator.Generate(indexArgument, dataArgument);

		// Finally, we end the order

		this->m_builder.AddStatement(endLabel);
	}
};

}
