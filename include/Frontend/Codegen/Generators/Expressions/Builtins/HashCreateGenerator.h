#pragma once

#include <limits>

#include "Frontend/Codegen/Generators/Generator.h"
#include "HorseIR/Traversal/ConstVisitor.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Data/TargetGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/ConversionGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/OperandGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/Builtins/InternalHashGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/AddressGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataSizeGenerator.h"
#include "Frontend/Codegen/NameUtils.h"

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

template<PTX::Bits B>
class HashCreateInsertGenerator : public Generator, public HorseIR::ConstVisitor
{
public:
	using Generator::Generator;

	std::string Name() const override { return "HashCreateInsertGenerator"; }

	void Generate(const HorseIR::Operand *dataOperand, PTX::TypedOperand<PTX::UInt32Type> *slot)
	{
		m_slot = slot;
		dataOperand->Accept(*this);
	}

	void Visit(const HorseIR::Identifier *identifier) override
	{
		DispatchType(*this, identifier->GetType(), identifier);
	}

	void Visit(const HorseIR::Literal *literal) override
	{
		Error("literal data for @hash_create");
	}

	template<class T>
	void GenerateVector(const HorseIR::Identifier *identifier)
	{
		// Nothing to do, CAS already inserted value
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
	void GenerateTuple(unsigned int index, const HorseIR::Operand *identifier)
	{
		if (index > 0)
		{
			GenerateStore<T>(identifier, index);
		}
	}

private:
	template<class T>
	void GenerateStore(const HorseIR::Operand *dataOperand, unsigned int cellIndex)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			//TODO: Smaller types
			Error("unimplemented");
		}
		else
		{
			// Load the value

			OperandGenerator<B, T> operandGenerator(this->m_builder);
			operandGenerator.SetBoundsCheck(false);
			auto value = operandGenerator.GenerateRegister(dataOperand, OperandGenerator<B, T>::LoadKind::Vector, cellIndex);

			// Get the kernel parameter (we assume it is always a cell), and compute the address

			auto kernelResources = this->m_builder.GetKernelResources();
			auto kernelParameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>>(NameUtils::ReturnName(0));

			AddressGenerator<B, T> valueAddressGenerator(this->m_builder);
			auto valueAddress = valueAddressGenerator.GenerateAddress(kernelParameter, cellIndex, m_slot);

			// Store!

			this->m_builder.AddStatement(new PTX::StoreInstruction<B, T, PTX::GlobalSpace>(valueAddress, value));
		}
	}

	PTX::TypedOperand<PTX::UInt32Type> *m_slot = nullptr;
};

template<PTX::Bits B>
class HashCreateGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "HashCreateGenerator"; }

	void Generate(const std::vector<const HorseIR::LValue *>& targets, const std::vector<const HorseIR::Operand *>& arguments)
	{
		auto dataArgument = arguments.at(0);
		DispatchType(*this, dataArgument->GetType(), dataArgument);
	}
	
	template<class T>
	void GenerateVector(const HorseIR::Operand *operand)
	{
		GenerateHashInsert<T>(operand);
	}

	template<class T>
	void GenerateList(const HorseIR::Operand *operand)
	{
		GenerateHashInsert<T>(operand, true);
	}
	
	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::Operand *operand)
	{
		if (index == 0)
		{
			GenerateHashInsert<T>(operand, true);
		}
	}

private:
	template<class T>
	void GenerateHashInsert(const HorseIR::Operand *operand, bool isCell = false)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value || std::is_same<T, PTX::Int8Type>::value)
		{
			//TODO: Smaller types
			Error("Unimplemented");
		}
		else
		{
			auto resources = this->m_builder.GetLocalResources();
			auto kernelResources = this->m_builder.GetKernelResources();

			DataIndexGenerator<B> indexGenerator(this->m_builder);
			auto index = indexGenerator.GenerateDataIndex();

			DataSizeGenerator<B> sizeGenerator(this->m_builder);
			auto dataSize = sizeGenerator.GenerateSize(operand);

			this->m_builder.AddIfStatement("HASH_SKIP", [&]()
			{
				auto sizePredicate = resources->template AllocateTemporary<PTX::PredicateType>();
				this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::UInt32Type>(
					sizePredicate, index, dataSize, PTX::UInt32Type::ComparisonOperator::GreaterEqual
				));
				return std::make_tuple(sizePredicate, false);
			},
			[&]()
			{
				ParameterGenerator<B> parameterGenerator(this->m_builder);
				auto capacityParameter = parameterGenerator.template GenerateConstant<PTX::UInt32Type>(NameUtils::HashtableSize);

				ValueLoadGenerator<B, PTX::UInt32Type> valueLoadGenerator(this->m_builder);
				auto capacity = valueLoadGenerator.GenerateConstant(capacityParameter);

				auto capacityM1 = resources->template AllocateTemporary<PTX::UInt32Type>();
				this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(capacityM1, capacity, new PTX::UInt32Value(1)));

				InternalHashGenerator<B> hashGenerator(this->m_builder);
				auto slot = hashGenerator.Generate(operand);
				auto currentSlot = resources->template AllocateTemporary<PTX::UInt32Type>();

				this->m_builder.AddDoWhileLoop("HASH", [&](Builder::LoopContext& loopContext)
				{
					this->m_builder.AddStatement(new PTX::AndInstruction<PTX::Bit32Type>(
						new PTX::Bit32RegisterAdapter<PTX::UIntType>(slot),
						new PTX::Bit32RegisterAdapter<PTX::UIntType>(slot),
						new PTX::Bit32RegisterAdapter<PTX::UIntType>(capacityM1)
					));

					using BitType = PTX::BitType<T::TypeBits>;

					auto empty = new PTX::Value<T>(std::numeric_limits<typename T::SystemType>::max());
					auto emptyRegister = resources->template AllocateTemporary<BitType>();
					this->m_builder.AddStatement(new PTX::MoveInstruction<BitType>(emptyRegister, new PTX::BitAdapter(empty)));

					// Check if the slot is already occupied

					OperandGenerator<B, BitType> operandGenerator(this->m_builder);
					operandGenerator.SetBoundsCheck(false);
					auto value = operandGenerator.GenerateOperand(operand, OperandGenerator<B, BitType>::LoadKind::Vector);

					// Get the address for the slot

					PTX::Address<B, T, PTX::GlobalSpace> *address = nullptr;
					if (isCell)
					{
						auto keyParameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::PointerType<B, T, PTX::GlobalSpace>>>(NameUtils::ReturnName(0));
						AddressGenerator<B, T> addressGenerator(this->m_builder);
						address = addressGenerator.GenerateAddress(keyParameter, 0, slot);
					}
					else
					{
						auto keyParameter = kernelResources->template GetParameter<PTX::PointerType<B, T>>(NameUtils::ReturnName(0));
						AddressGenerator<B, T> addressGenerator(this->m_builder);
						address = addressGenerator.GenerateAddress(keyParameter, slot);
					}

					auto bitAddress = new PTX::AddressAdapter<B, BitType, T, PTX::GlobalSpace>(address);

					// Increment before jumping, so we can make the exit a straight shot

					this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::UInt32Type>(currentSlot, slot));
					this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(slot, slot, new PTX::UInt32Value(1)));

					// Precheck value for empty, as CAS is expensive

					auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
					auto previous = resources->template AllocateTemporary<BitType>();

					this->m_builder.AddStatement(new PTX::LoadInstruction<B, BitType, PTX::GlobalSpace>(previous, bitAddress));
					this->m_builder.AddStatement(new PTX::SetPredicateInstruction<BitType>(
						predicate, previous, emptyRegister, BitType::ComparisonOperator::Equal
					));

					// CAS!

					auto atomicInstruction = new PTX::AtomicInstruction<B, BitType, PTX::GlobalSpace>(
						previous, bitAddress, emptyRegister, value, BitType::AtomicOperation::CompareAndSwap
					);
					atomicInstruction->SetPredicate(predicate);
					this->m_builder.AddStatement(atomicInstruction);

					this->m_builder.AddStatement(new PTX::SetPredicateInstruction<BitType>(
						predicate, previous, emptyRegister, BitType::ComparisonOperator::NotEqual
					));
					return std::make_tuple(predicate, false);
				});

				// Insert other values

				HashCreateInsertGenerator<B> insertGenerator(this->m_builder);
				insertGenerator.Generate(operand, currentSlot);

				// Store value (index)

				auto valueParameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::Int64Type>>(NameUtils::ReturnName(1));

				AddressGenerator<B, PTX::Int64Type> valueAddressGenerator(this->m_builder);
				auto valueAddress = valueAddressGenerator.GenerateAddress(valueParameter, currentSlot);

				DataIndexGenerator<B> indexGenerator(this->m_builder);
				auto index = indexGenerator.GenerateDataIndex();
				auto index64 = ConversionGenerator::ConvertSource<PTX::Int64Type, PTX::UInt32Type>(this->m_builder, index);

				this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::Int64Type, PTX::GlobalSpace>(valueAddress, index64));
			});
		}
	}
};

}
}
