#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"
#include "Codegen/Generators/Expressions/ConversionGenerator.h"

#include "PTX/PTX.h"

namespace Codegen {

template<PTX::Bits B>
class AtomicGenerator : public Generator
{
public:
	using Generator::Generator;

	std::string Name() const override { return "AtomicGenerator"; }

	void GenerateWait(const PTX::GlobalVariable<PTX::Bit32Type> *lock)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Generate a loop, waiting on the lock (value 1=held, value 0=free)
		//
		// ATOM:
		//   atom.global.cas.b32 %value, [lock], 0, 1
		//   setp.ne.b32 %p, %value, 0
		//   @%p bra ATOM

		auto label = this->m_builder.CreateLabel("ATOM");
		this->m_builder.AddStatement(label);

		auto value = resources->template AllocateTemporary<PTX::Bit32Type>();
		auto predicate = resources->template AllocateTemporary<PTX::PredicateType>();

		auto lockAddress = new PTX::MemoryAddress<B, PTX::Bit32Type, PTX::GlobalSpace>(lock);

		this->m_builder.AddStatement(new PTX::AtomicInstruction<B, PTX::Bit32Type, PTX::GlobalSpace, PTX::Bit32Type::AtomicOperation::CompareAndSwap>(
			value, lockAddress, new PTX::Bit32Adapter<PTX::UIntType>(new PTX::UInt32Value(0)), new PTX::Bit32Adapter<PTX::UIntType>(new PTX::UInt32Value(1))
		));

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<PTX::Bit32Type>(
			predicate, value, new PTX::Bit32Adapter<PTX::UIntType>(new PTX::UInt32Value(0)), PTX::Bit32Type::ComparisonOperator::NotEqual
		));
		this->m_builder.AddStatement(new PTX::BranchInstruction(label, predicate));
	}

	void GenerateUnlock(const PTX::GlobalVariable<PTX::Bit32Type> *lock)
	{
		auto resources = this->m_builder.GetLocalResources();

		// Release the lock by setting the value to 0

		auto value = resources->template AllocateTemporary<PTX::Bit32Type>();
		auto lockAddress = new PTX::MemoryAddress<B, PTX::Bit32Type, PTX::GlobalSpace>(lock);

		this->m_builder.AddStatement(new PTX::MoveInstruction<PTX::Bit32Type>(value, new PTX::Bit32Adapter<PTX::UIntType>(new PTX::UInt32Value(0))));
		this->m_builder.AddStatement(new PTX::StoreInstruction<B, PTX::Bit32Type, PTX::GlobalSpace>(lockAddress, value));
	}

	template<class T>
	void GenerateMinMaxReduction(const PTX::Address<B, T, PTX::GlobalSpace> *globalAddress, const PTX::TypedOperand<T> *value, bool min)
	{
		auto resources = this->m_builder.GetLocalResources();
		using BitType = PTX::BitType<T::TypeBits>;

		// Generate a loop, at each iteration checking if the input value is less than the global.
		// If so, try a replacement until either: (1) the value is no longer minimal or (2) the replacement succeeds
		//
		//    setp.ge.T %p1, %val, %global;
		//    @%p1 bra END;
		//    mov.T %old, %global;
		//
		// START:
		//    mov.T 	%assumed, %old;
		//    atom.global.cas.Bit<T> %old, [%global_address], %assumed, %val;
		//
		//    setp.ne.T %p2, %assumed, %old;
		//    setp.lt.T %p3, %val, %old;
		//    and.pred  %p4, %p2, %p3;
		//    @%p4 bra START;
		//
		// END:

		auto old = resources->template AllocateTemporary<T>();
		this->m_builder.AddStatement(new PTX::LoadInstruction<B, T, PTX::GlobalSpace>(old, globalAddress));

		auto startLabel = this->m_builder.CreateLabel("START");
		auto endLabel = this->m_builder.CreateLabel("END");
		auto predicate0 = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(predicate0, value, old, (min) ? T::ComparisonOperator::GreaterEqual : T::ComparisonOperator::LessEqual));
		this->m_builder.AddStatement(new PTX::BranchInstruction(endLabel, predicate0));

		this->m_builder.AddStatement(startLabel);

		auto assumed = resources->template AllocateTemporary<T>();
		this->m_builder.AddStatement(new PTX::MoveInstruction<T>(assumed, old));

		auto bitOld = ConversionGenerator::ConvertSource<BitType, T>(this->m_builder, old);
		auto bitAddress = new PTX::AddressAdapter<B, BitType, T, PTX::GlobalSpace>(globalAddress);
		auto bitAssumed = ConversionGenerator::ConvertSource<BitType, T>(this->m_builder, assumed);
		auto bitValue = ConversionGenerator::ConvertSource<BitType, T>(this->m_builder, value);

		this->m_builder.AddStatement(new PTX::AtomicInstruction<B, BitType, PTX::GlobalSpace, BitType::AtomicOperation::CompareAndSwap>(
			bitOld, bitAddress, bitAssumed, bitValue
		));

		auto predicate1 = resources->template AllocateTemporary<PTX::PredicateType>();
		auto predicate2 = resources->template AllocateTemporary<PTX::PredicateType>();
		auto predicate3 = resources->template AllocateTemporary<PTX::PredicateType>();

		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(predicate1, assumed, old, T::ComparisonOperator::NotEqual));
		this->m_builder.AddStatement(new PTX::SetPredicateInstruction<T>(predicate2, value, old, (min) ? T::ComparisonOperator::Less : T::ComparisonOperator::Greater));
		this->m_builder.AddStatement(new PTX::AndInstruction<PTX::PredicateType>(predicate3, predicate1, predicate2));
		this->m_builder.AddStatement(new PTX::BranchInstruction(startLabel, predicate3));

		this->m_builder.AddStatement(endLabel);
	}
};

}
