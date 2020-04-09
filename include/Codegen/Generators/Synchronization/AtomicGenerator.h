#pragma once

#include "Codegen/Generators/Generator.h"

#include "Codegen/Builder.h"

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
};

}
