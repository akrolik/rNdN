#include "Backend/Codegen/Generators/Instructions/Data/LoadGenerator.h"

#include "Backend/Codegen/Generators/Operands/AddressGenerator.h"
#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

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
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types: *, Vector (exclude Float16(x2))
	// Spaces: *
	// Modifiers: --

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<PTX::Bits B, class T, class S, PTX::LoadSynchronization A>
void LoadGenerator::GenerateMaxwell(const PTX::LoadInstruction<B, T, S, A> *instruction)
{
	// Verify permissible synchronization

	if constexpr(A == PTX::LoadSynchronization::Weak)
	{
		if constexpr(std::is_same<S, PTX::ParameterSpace>::value)
		{
			// Generate destination operand

			RegisterGenerator registerGenerator(this->m_builder);
			auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());

			// Generate address operand

			CompositeGenerator compositeGenerator(this->m_builder);
			auto [address_Lo, address_Hi] = compositeGenerator.GeneratePair(instruction->GetAddress());

			// Generate instruction

			this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination_Lo, address_Lo));

			// Extended datatypes

			if constexpr(T::TypeBits == PTX::Bits::Bits64)
			{
				this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination_Hi, address_Hi));
			}
		}
		else
		{
			// Generate destination operand

			RegisterGenerator registerGenerator(this->m_builder);
			auto destination = registerGenerator.Generate(instruction->GetDestination());

			// Generate address operand

			AddressGenerator addressGenerator(this->m_builder);
			auto address = addressGenerator.Generate(instruction->GetAddress());

			if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
			{
				// Generate instruction

				auto type = InstructionType<SASS::Maxwell::LDGInstruction::Type, T>();
				auto flags = SASS::Maxwell::LDGInstruction::Flags::None;
				if constexpr(B == PTX::Bits::Bits64)
				{
					flags |= SASS::Maxwell::LDGInstruction::Flags::E;
				}

				auto cache = SASS::Maxwell::LDGInstruction::Cache::None;
				switch (instruction->GetCacheOperator())
				{
					// case PTX::LoadInstruction<B, T, S, A>::CacheOperator::All:
					// case PTX::LoadInstruction<B, T, S, A>::CacheOperator::Streaming:
					// {
					// 	cache = SASS::Maxwell::LDGInstruction::Cache::None;
					// 	break;
					// }
					case PTX::LoadInstruction<B, T, S, A>::CacheOperator::Global:
					case PTX::LoadInstruction<B, T, S, A>::CacheOperator::LastUse:
					{
						cache = SASS::Maxwell::LDGInstruction::Cache::CG;
						break;
					}
					case PTX::LoadInstruction<B, T, S, A>::CacheOperator::Invalidate:
					{
						cache = SASS::Maxwell::LDGInstruction::Cache::CV;
						break;
					}
				}

				this->AddInstruction(new SASS::Maxwell::LDGInstruction(destination, address, type, cache, flags));
			}
			else if constexpr(std::is_same<S, PTX::SharedSpace>::value)
			{
				// Generate instruction

				auto type = InstructionType<SASS::Maxwell::LDSInstruction::Type, T>();
				auto flags = SASS::Maxwell::LDSInstruction::Flags::None;

				// Flag not necessary for shared variables
				//
				// if constexpr(B == PTX::Bits::Bits64)
				// {
				// 	flags |= SASS::Maxwell::LDSInstruction::Flags::E;
				// }

				this->AddInstruction(new SASS::Maxwell::LDSInstruction(destination, address, type, flags));
			}
			else
			{
				Error(instruction, "unsupported space");
			}
		}
	}
	else
	{
		Error(instruction, "unsupported synchronzation modifier");
	}
}

template<PTX::Bits B, class T, class S, PTX::LoadSynchronization A>
void LoadGenerator::GenerateVolta(const PTX::LoadInstruction<B, T, S, A> *instruction)
{
	Error(instruction, "unsupported architecture");
}

}
}
