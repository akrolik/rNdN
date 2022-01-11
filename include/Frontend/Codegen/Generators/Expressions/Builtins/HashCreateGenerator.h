#pragma once

#include <limits>

#include "Frontend/Codegen/Generators/Generator.h"
#include "HorseIR/Traversal/ConstVisitor.h"

#include "Frontend/Codegen/Builder.h"
#include "Frontend/Codegen/Generators/Data/TargetGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/ConversionGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/MoveGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/OperandGenerator.h"
#include "Frontend/Codegen/Generators/Expressions/Builtins/InternalHashGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/AddressGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataIndexGenerator.h"
#include "Frontend/Codegen/Generators/Indexing/DataSizeGenerator.h"
#include "Frontend/Codegen/Generators/Synchronization/AtomicGenerator.h"
#include "Frontend/Codegen/NameUtils.h"

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeUtils.h"

#include "HorseIR/Tree/Tree.h"

#include "PTX/Tree/Tree.h"

namespace Frontend {
namespace Codegen {

// Inspired by: https://github.com/nosferalatu/SimpleGPUHashTable

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
			GenerateStore<PTX::Int8Type>(dataOperand, cellIndex);
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

	void Generate(const std::vector<const HorseIR::LValue *>& targets, const std::vector<const HorseIR::Operand *>& arguments, bool storeValue = true)
	{
		std::vector<const HorseIR::Operand *> functionArguments(std::begin(arguments), std::end(arguments) - 1);
		auto dataArgument = arguments.back();

		std::vector<ComparisonOperation> joinOperations;
		if (functionArguments.size() == 0)
		{
			joinOperations.push_back(ComparisonOperation::Equal);
		}
		else
		{
			for (auto functionArgument : functionArguments)
			{
				if (auto functionType = HorseIR::TypeUtils::GetType<HorseIR::FunctionType>(functionArgument->GetType()))
				{
					auto joinOperation = GetJoinComparisonOperation(functionType->GetFunctionDeclaration(), true);
					joinOperations.push_back(joinOperation);
				}
				else
				{
					Generator::Error("non-function join argument '" + HorseIR::PrettyPrinter::PrettyString(functionArgument, true) + "'");
				}
			}
		}
		m_storeValue = storeValue;

		DispatchType(*this, dataArgument->GetType(), dataArgument, joinOperations);
	}
	
	template<class T>
	void GenerateVector(const HorseIR::Operand *operand, const std::vector<ComparisonOperation>& joinOperations)
	{
		GenerateHashInsert<T>(operand, joinOperations);
	}

	template<class T>
	void GenerateList(const HorseIR::Operand *operand, const std::vector<ComparisonOperation>& joinOperations)
	{
		GenerateHashInsert<T>(operand, joinOperations, true);
	}
	
	template<class T>
	void GenerateTuple(unsigned int index, const HorseIR::Operand *operand, const std::vector<ComparisonOperation>& joinOperations)
	{
		if (index == 0)
		{
			GenerateHashInsert<T>(operand, joinOperations, true);
		}
	}

private:
	template<class T>
	void GenerateHashInsert(const HorseIR::Operand *operand, const std::vector<ComparisonOperation>& joinOperations, bool isCell = false)
	{
		if constexpr(std::is_same<T, PTX::PredicateType>::value)
		{
			GenerateHashInsert<PTX::Int8Type>(operand, joinOperations, isCell);
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
				auto slot = hashGenerator.Generate(operand, joinOperations);

				auto empty = new PTX::Value<T>(std::numeric_limits<typename T::SystemType>::max());
				auto emptyRegister = resources->template AllocateTemporary<T>();

				MoveGenerator<T> moveGenerator(this->m_builder);
				moveGenerator.Generate(emptyRegister, empty);

				using BitType = PTX::BitType<T::TypeBits>;

				OperandGenerator<B, BitType> operandGenerator(this->m_builder);
				operandGenerator.SetBoundsCheck(false);
				auto value = operandGenerator.GenerateRegister(operand, OperandGenerator<B, BitType>::LoadKind::Vector);

				this->m_builder.AddDoWhileLoop("HASH", [&](Builder::LoopContext& loopContext)
				{
					// Keep within bounds

					this->m_builder.AddStatement(new PTX::AndInstruction<PTX::Bit32Type>(
						new PTX::Bit32RegisterAdapter<PTX::UIntType>(slot),
						new PTX::Bit32RegisterAdapter<PTX::UIntType>(slot),
						new PTX::Bit32RegisterAdapter<PTX::UIntType>(capacityM1)
					));

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

					// Precheck value for empty, as CAS is expensive

					auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();
					auto previous = resources->template AllocateTemporary<T>();

					this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::GlobalSpace>(previous, address));

					ComparisonGenerator<B, PTX::PredicateType> comparisonGeneratorEQ(this->m_builder, ComparisonOperation::Equal);
					comparisonGeneratorEQ.template Generate<T>(predicate, previous, emptyRegister);

					// CAS!

					auto bitAddress = new PTX::AddressAdapter<B, BitType, T, PTX::GlobalSpace>(address);

					if constexpr(std::is_same<T, PTX::Int8Type>::value)
					{
						this->m_builder.AddIfStatement("SLOT_SKIP", [&]()
						{
							return std::make_tuple(predicate, true);
						},
						[&]()
						{
							// Generate lock variable and lock

							auto globalResources = this->m_builder.GetGlobalResources();
							auto lock = globalResources->template AllocateGlobalVariable<PTX::Bit32Type>(this->m_builder.UniqueIdentifier("hash_lock"));

							AtomicGenerator<B> atomicGenerator(this->m_builder);
							atomicGenerator.GenerateWait(lock);

							// Load the existing value

							this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::GlobalSpace>(previous, address));

							// Check if slot still empty

							ComparisonGenerator<B, PTX::PredicateType> comparisonGeneratorEQ(this->m_builder, ComparisonOperation::Equal);
							comparisonGeneratorEQ.template Generate<T>(predicate, previous, emptyRegister);

							// If empty, store the current value

							auto store = new PTX::StoreInstruction<B, BitType, PTX::GlobalSpace>(bitAddress, value);
							store->SetPredicate(predicate);
							this->m_builder.AddStatement(store);

							// Unlock!

							atomicGenerator.GenerateUnlock(lock);
						});
					}
					else
					{
						auto bitEmpty = new PTX::VariableAdapter<BitType, T, PTX::RegisterSpace>(emptyRegister);
						auto bitPrevious = new PTX::VariableAdapter<BitType, T, PTX::RegisterSpace>(previous);

						auto atomicInstruction = new PTX::AtomicInstruction<B, BitType, PTX::GlobalSpace>(
							bitPrevious, bitAddress, bitEmpty, value, BitType::AtomicOperation::CompareAndSwap
						);
						atomicInstruction->SetPredicate(predicate);
						this->m_builder.AddStatement(atomicInstruction);
					}

					ComparisonGenerator<B, PTX::PredicateType> comparisonGeneratorNE(this->m_builder, ComparisonOperation::NotEqual);
					comparisonGeneratorNE.template Generate<T>(predicate, previous, emptyRegister);

					// Increment before jumping, so we can make the exit a straight shot

					this->m_builder.AddStatement(new PTX::AddInstruction<PTX::UInt32Type>(slot, slot, new PTX::UInt32Value(1)));

					return std::make_tuple(predicate, false);
				});

				// Insert other values

				this->m_builder.AddStatement(new PTX::SubtractInstruction<PTX::UInt32Type>(slot, slot, new PTX::UInt32Value(1)));

				HashCreateInsertGenerator<B> insertGenerator(this->m_builder);
				insertGenerator.Generate(operand, slot);

				if (m_storeValue)
				{
					// Store value (index)

					auto valueParameter = kernelResources->template GetParameter<PTX::PointerType<B, PTX::Int64Type>>(NameUtils::ReturnName(1));

					AddressGenerator<B, PTX::Int64Type> valueAddressGenerator(this->m_builder);
					auto valueAddress = valueAddressGenerator.GenerateAddress(valueParameter, slot);

					DataIndexGenerator<B> indexGenerator(this->m_builder);
					auto index = indexGenerator.GenerateDataIndex();
					auto index64 = ConversionGenerator::ConvertSource<PTX::Int64Type, PTX::UInt32Type>(this->m_builder, index);

					this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::Int64Type, PTX::GlobalSpace>(valueAddress, index64));
				}
			});
		}
	}

	bool m_storeValue = true;
};

}
}
