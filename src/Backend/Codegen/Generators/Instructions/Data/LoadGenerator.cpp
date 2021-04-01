#include "Backend/Codegen/Generators/Instructions/Data/LoadGenerator.h"

#include "Backend/Codegen/Generators/Operands/AddressGenerator.h"
#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void LoadGenerator::Generate(const PTX::_LoadInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<typename I, class T>
I LoadGenerator::InstructionType()
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
	Error("load for type " + T::Name());
}

template<PTX::Bits B, class T, class S, PTX::LoadSynchronization A>
void LoadGenerator::Visit(const PTX::LoadInstruction<B, T, S, A> *instruction)
{
	// Types: *, Vector (exclude Float16(x2))
	// Spaces: *
	// Modifiers: --

	// Verify permissible synchronization

	if constexpr(A == PTX::LoadSynchronization::Weak)
	{
		// Generate destination operand

		RegisterGenerator registerGenerator(this->m_builder);
		auto [destination, destination_Hi] = registerGenerator.Generate(instruction->GetDestination());

		if constexpr(std::is_same<S, PTX::ParameterSpace>::value)
		{
			CompositeGenerator compositeGenerator(this->m_builder);
			auto [address, address_Hi] = compositeGenerator.Generate(instruction->GetAddress());

			// Generate instruction

			this->AddInstruction(new SASS::MOVInstruction(destination, address));

			// Extended datatypes

			if constexpr(T::TypeBits == PTX::Bits::Bits64)
			{
				this->AddInstruction(new SASS::MOVInstruction(destination_Hi, address_Hi));
			}
		}
		else
		{
			// Generate address operand

			AddressGenerator addressGenerator(this->m_builder);
			auto address = addressGenerator.Generate(instruction->GetAddress());

			if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
			{
				// Generate instruction

				auto type = InstructionType<SASS::LDGInstruction::Type, T>();
				auto flags = SASS::LDGInstruction::Flags::None;
				if constexpr(B == PTX::Bits::Bits64)
				{
					flags |= SASS::LDGInstruction::Flags::E;
				}

				auto cache = SASS::LDGInstruction::Cache::None;
				switch (instruction->GetCacheOperator())
				{
					// case PTX::LoadInstruction<B, T, S, A>::CacheOperator::All:
					// case PTX::LoadInstruction<B, T, S, A>::CacheOperator::Streaming:
					// {
					// 	cache = SASS::LDGInstruction::Cache::None;
					// 	break;
					// }
					case PTX::LoadInstruction<B, T, S, A>::CacheOperator::Global:
					case PTX::LoadInstruction<B, T, S, A>::CacheOperator::LastUse:
					{
						cache = SASS::LDGInstruction::Cache::CG;
						break;
					}
					case PTX::LoadInstruction<B, T, S, A>::CacheOperator::Invalidate:
					{
						cache = SASS::LDGInstruction::Cache::CV;
						break;
					}
				}

				this->AddInstruction(new SASS::LDGInstruction(destination, address, type, cache, flags));
			}
			else if constexpr(std::is_same<S, PTX::SharedSpace>::value)
			{
				// Generate instruction

				auto type = InstructionType<SASS::LDSInstruction::Type, T>();
				auto flags = SASS::LDSInstruction::Flags::None;

				// Flag not necessary for shared variables
				//
				// if constexpr(B == PTX::Bits::Bits64)
				// {
				// 	flags |= SASS::LDSInstruction::Flags::E;
				// }

				this->AddInstruction(new SASS::LDSInstruction(destination, address, type, flags));
			}
			else
			{
				Error(instruction, "unsupported space");
			}

			this->AddInstruction(new SASS::DEPBARInstruction(
				SASS::DEPBARInstruction::Barrier::SB0, new SASS::I8Immediate(0x0), SASS::DEPBARInstruction::Flags::LE
			));
		}
	}
	else
	{
		Error(instruction, "unsupported synchronzation modifier");
	}

}

}
}
