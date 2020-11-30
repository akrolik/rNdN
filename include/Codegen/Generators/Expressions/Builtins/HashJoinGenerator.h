#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Data/TargetCellGenerator.h"
#include "Codegen/Generators/Expressions/Builtins/ComparisonGenerator.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Codegen {

template<PTX::Bits B>
class HashJoinGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "HashJoinGenerator"; }

	void Generate(const HorseIR::LValue *target, const std::vector<HorseIR::Operand *>& arguments)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Decompose arguments

		auto keyArgument = arguments.at(0);
		auto dataArgument = arguments.at(2);
		auto offsetArgument = arguments.at(3);

		// Bounds check

		auto startLabel = this->m_builder.CreateLabel("START");
		auto endLabel = this->m_builder.CreateLabel("END");

		DataIndexGenerator<B> indexGenerator(this->m_builder);
		auto index = indexGenerator.GenerateDataIndex();

		DataSizeGenerator<B> sizeGenerator(this->m_builder);
		auto dataSize = sizeGenerator.GenerateSize(dataArgument);
		auto sizePredicate = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(sizePredicate, index, dataSize, PTX::UInt32Type::ComparisonOperator::GreaterEqual));
		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel, sizePredicate));

		// Get write offset

		OperandGenerator<B, PTX::Int64Type> operandGenerator(this->m_builder);
		operandGenerator.SetBoundsCheck(false);
		auto writeOffset = operandGenerator.GenerateOperand(offsetArgument, OperandGenerator<B, PTX::Int64Type>::LoadKind::Vector);

		m_writeOffset = resources->template AllocateTemporary<PTX::Int64Type>();
		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Int64Type>(m_writeOffset, writeOffset));

		auto capacity = sizeGenerator.GenerateSize(keyArgument);
		auto capacityM1 = resources->template AllocateTemporary<PTX::UInt32Type>();
		this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(capacityM1, capacity, new PTX::UInt32Value(1)));

		// Compute the hash value

		InternalHashGenerator<B> hashGenerator(this->m_builder);
		auto slot = hashGenerator.Generate(dataArgument);

		// Keep the slot within range

		this->m_builder.AddStatement(startLabel);
		this->m_builder.AddStatement(new PTX::AndInstruction<PTX::Bit32Type>(
			new PTX::Bit32RegisterAdapter<PTX::UIntType>(slot),
			new PTX::Bit32RegisterAdapter<PTX::UIntType>(slot),
			new PTX::Bit32RegisterAdapter<PTX::UIntType>(capacityM1)
		));

		DispatchType(*this, dataArgument->GetType(), arguments, slot, startLabel);

		this->m_builder.AddStatement(endLabel);

	}

	template<class T>
	void GenerateVector(const std::vector<HorseIR::Operand *>& arguments, const PTX::Register<PTX::UInt32Type> *slot, const PTX::Label *startLabel)
	{
		GenerateHashLookup<T>(arguments, slot, startLabel);
	}

	template<class T>
	void GenerateList(const std::vector<HorseIR::Operand *>& arguments, const PTX::Register<PTX::UInt32Type> *slot, const PTX::Label *startLabel)
	{
		GenerateHashLookup<T>(arguments, slot, startLabel);
	}
	
	template<class T>
	void GenerateTuple(unsigned int index, const std::vector<HorseIR::Operand *>& arguments, const PTX::Register<PTX::UInt32Type> *slot, const PTX::Label *startLabel)
	{
		if (index == 0)
		{
			GenerateHashLookup<T>(arguments, slot, startLabel);
		}
	}

private:
	template<class T>
	void GenerateHashLookup(const std::vector<HorseIR::Operand *>& arguments, const PTX::Register<PTX::UInt32Type> *slot, const PTX::Label *startLabel)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value || std::is_same<T, PTX::Int8Type>::value)
		{
			//TODO: Smaller types
			Error("Unimplemented");
		}
		else
		{
			auto keyOperand = arguments.at(0);
			auto valueOperand = arguments.at(1);
			auto dataOperand = arguments.at(2);

			auto resources = this->m_builder.GetLocalResources();
			auto emptyPredicate = resources->template AllocateTemporary<PTX::PredicateType>();
			auto elseLabel = this->m_builder.CreateLabel("ELSE");

			InternalHashEqualGenerator<B, T> equalGenerator(this->m_builder);
			auto [equalPredicate, slotValue] = equalGenerator.Generate(dataOperand, keyOperand, slot);

			auto oldSlot = resources->template AllocateTemporary<PTX::UInt32Type>();
			this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(oldSlot, slot));
			this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(slot, slot, new PTX::UInt32Value(1)));

			this->m_builder.AddStatement(new PTX::BranchInstruction(elseLabel, equalPredicate, true));

			// Output all indexes, requires breaking the typical return pattern as there is an indeterminate quantity

			auto kernelResources = this->m_builder.GetKernelResources();
			auto returnParameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::PointerType<B, PTX::Int64Type, PTX::GlobalSpace>>>(NameUtils::ReturnName(0));

			//TODO: Conversions need to be optimized. Likely by a consistent index size for all data

			auto writeOffset = ConversionGenerator::ConvertSource<PTX::UInt32Type, PTX::Int64Type>(this->m_builder, m_writeOffset);

			AddressGenerator<B, PTX::Int64Type> addressGenerator(this->m_builder);
			auto returnAddress0 = addressGenerator.GenerateAddress(returnParameter, 0, writeOffset);
			auto returnAddress1 = addressGenerator.GenerateAddress(returnParameter, 1, writeOffset);

			ThreadIndexGenerator<B> indexGenerator(this->m_builder);
			auto globalIndex = indexGenerator.GenerateGlobalIndex();

			auto globalIndex64 = ConversionGenerator::ConvertSource<PTX::Int64Type, PTX::UInt32Type>(this->m_builder, globalIndex);

			OperandGenerator<B, PTX::Int64Type> operandGenerator64(this->m_builder);
			operandGenerator64.SetBoundsCheck(false);
			auto matchValue = operandGenerator64.GenerateRegister(valueOperand, oldSlot, this->m_builder.UniqueIdentifier("slot"));

			this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::Int64Type, PTX::GlobalSpace>(returnAddress0, matchValue));
			this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::Int64Type, PTX::GlobalSpace>(returnAddress1, globalIndex64));

			// End control flow and increment both the matched (predicated) and running counts

			this->m_builder.AddStatement(new PTX::AddInstruction<PTX::Int64Type>(m_writeOffset, m_writeOffset, new PTX::Int64Value(1)));
			this->m_builder.AddStatement(new PTX::BranchInstruction(startLabel));

			// No match, check for empty

			this->m_builder.AddStatement(elseLabel);

			auto empty = new PTX::Value<T>(std::numeric_limits<typename T::SystemType>::max());
			this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(emptyPredicate, slotValue, empty, T::ComparisonOperator::NotEqual));
			this->m_builder.AddStatement(new PTX::BranchInstruction(startLabel, emptyPredicate));
		}
	}

	const PTX::Register<PTX::Int64Type> *m_writeOffset = nullptr;
};

}
