#include "Backend/Codegen/Generators/Instructions/Data/StoreGenerator.h"

#include "Backend/Codegen/Generators/Operands/AddressGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void StoreGenerator::Generate(const PTX::_StoreInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<typename I, class T>
I StoreGenerator::InstructionType()
{
	if constexpr(std::is_same<T, PTX::UInt8Type>::value || std::is_same<T, PTX::Bit8Type>::value)
	{
		return I::U8;
	}
	else if constexpr(std::is_same<T, PTX::Int8Type>::value)
	{
		return I::S8;
	}
	else if constexpr(std::is_same<T, PTX::UInt16Type>::value || std::is_same<T, PTX::Bit16Type>::value)
	{
		return I::U16;
	}
	else if constexpr(std::is_same<T, PTX::Int16Type>::value)
	{
		return I::S16;
	}
	else if constexpr(T::TypeBits == PTX::Bits::Bits32)
	{
		return I::X32;
	}
	else if constexpr(T::TypeBits == PTX::Bits::Bits64)
	{
		return I::X64;
	}
	Error("store for type " + T::Name());
}

template<PTX::Bits B, class T, class S, PTX::StoreSynchronization A>
void StoreGenerator::Visit(const PTX::StoreInstruction<B, T, S, A> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types: *, Vector (exclude Float16(x2))
	// Spaces: *
	// Modifiers: --

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<PTX::Bits B, class T, class S, PTX::StoreSynchronization A>
void StoreGenerator::GenerateMaxwell(const PTX::StoreInstruction<B, T, S, A> *instruction)
{
	// Verify permissible synchronization

	if constexpr(A == PTX::StoreSynchronization::Weak)
	{
		// Generate source operand

		RegisterGenerator registerGenerator(this->m_builder);
		auto source = registerGenerator.Generate(instruction->GetSource());

		// Generate address operand

		AddressGenerator addressGenerator(this->m_builder);
		auto address = addressGenerator.Generate(instruction->GetAddress());

		if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
		{
			// Generate instruction

			auto type = InstructionType<SASS::Maxwell::STGInstruction::Type, T>();
			auto flags = SASS::Maxwell::STGInstruction::Flags::None;
			if constexpr(B == PTX::Bits::Bits64)
			{
				flags = SASS::Maxwell::STGInstruction::Flags::E;
			}

			auto cache = SASS::Maxwell::STGInstruction::Cache::None;
			switch (instruction->GetCacheOperator())
			{
				// case PTX::StoreInstruction<B, T, S, A>::CacheOperator::WriteBack:
				// {
				// 	cache = SASS::Maxwell::STGInstruction::Cache::None;
				// 	break;
				// }
				case PTX::StoreInstruction<B, T, S, A>::CacheOperator::Global:
				{
					cache = SASS::Maxwell::STGInstruction::Cache::CG;
					break;
				}
				case PTX::StoreInstruction<B, T, S, A>::CacheOperator::Streaming:
				{
					cache = SASS::Maxwell::STGInstruction::Cache::CS;
					break;
				}
				case PTX::StoreInstruction<B, T, S, A>::CacheOperator::WriteThrough:
				{
					cache = SASS::Maxwell::STGInstruction::Cache::WT;
					break;
				}
			}

			this->AddInstruction(new SASS::Maxwell::STGInstruction(address, source, type, cache, flags));
		}
		else if constexpr(std::is_same<S, PTX::SharedSpace>::value)
		{
			// Generate instruction

			auto type = InstructionType<SASS::Maxwell::STSInstruction::Type, T>();
			auto flags = SASS::Maxwell::STSInstruction::Flags::None;

			// Flag not necessary for shared variables
			//
			// if constexpr(B == PTX::Bits::Bits64)
			// {
			// 	flags |= SASS::Maxwell::STSInstruction::Flags::E;
			// }

			this->AddInstruction(new SASS::Maxwell::STSInstruction(address, source, type, flags));
		}
		else
		{
			Error(instruction, "unsupported space");
		}
	}
	else
	{
		Error(instruction, "unsupported synchronzation modifier");
	}
}

template<PTX::Bits B, class T, class S, PTX::StoreSynchronization A>
void StoreGenerator::GenerateVolta(const PTX::StoreInstruction<B, T, S, A> *instruction)
{
	Error(instruction, "unsupported architecture");
}

}
}
