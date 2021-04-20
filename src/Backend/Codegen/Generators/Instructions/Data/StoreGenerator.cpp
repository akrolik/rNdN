#include "Backend/Codegen/Generators/Instructions/Data/StoreGenerator.h"

#include "Backend/Codegen/Generators/Operands/AddressGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

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
	// Types: *, Vector (exclude Float16(x2))
	// Spaces: *
	// Modifiers: --

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

			auto type = InstructionType<SASS::STGInstruction::Type, T>();
			auto flags = SASS::STGInstruction::Flags::None;
			if constexpr(B == PTX::Bits::Bits64)
			{
				flags = SASS::STGInstruction::Flags::E;
			}

			auto cache = SASS::STGInstruction::Cache::None;
			switch (instruction->GetCacheOperator())
			{
				// case PTX::StoreInstruction<B, T, S, A>::CacheOperator::WriteBack:
				// {
				// 	cache = SASS::STGInstruction::Cache::None;
				// 	break;
				// }
				case PTX::StoreInstruction<B, T, S, A>::CacheOperator::Global:
				{
					cache = SASS::STGInstruction::Cache::CG;
					break;
				}
				case PTX::StoreInstruction<B, T, S, A>::CacheOperator::Streaming:
				{
					cache = SASS::STGInstruction::Cache::CS;
					break;
				}
				case PTX::StoreInstruction<B, T, S, A>::CacheOperator::WriteThrough:
				{
					cache = SASS::STGInstruction::Cache::WT;
					break;
				}
			}

			this->AddInstruction(new SASS::STGInstruction(address, source, type, cache, flags));
		}
		else if constexpr(std::is_same<S, PTX::SharedSpace>::value)
		{
			// Generate instruction

			auto type = InstructionType<SASS::STSInstruction::Type, T>();
			auto flags = SASS::STSInstruction::Flags::None;

			// Flag not necessary for shared variables
			//
			// if constexpr(B == PTX::Bits::Bits64)
			// {
			// 	flags |= SASS::STSInstruction::Flags::E;
			// }

			this->AddInstruction(new SASS::STSInstruction(address, source, type, flags));
		}
		else
		{
			Error(instruction, "unsupported space");
		}

		this->AddInstruction(new SASS::DEPBARInstruction(
			SASS::DEPBARInstruction::Barrier::SB0, new SASS::I8Immediate(0x0), SASS::DEPBARInstruction::Flags::LE
		));
	}
	else
	{
		Error(instruction, "unsupported synchronzation modifier");
	}
}

}
}
