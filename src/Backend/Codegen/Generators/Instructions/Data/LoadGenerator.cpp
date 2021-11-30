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

template<typename LDInstruction, class T>
typename LDInstruction::Type LoadGenerator::InstructionType()
{
	if constexpr(std::is_same<T, PTX::UInt8Type>::value || std::is_same<T, PTX::Bit8Type>::value)
	{
		return LDInstruction::Type::U8;
	}
	else if constexpr(std::is_same<T, PTX::Int8Type>::value)
	{
		return LDInstruction::Type::S8;
	}
	else if constexpr(std::is_same<T, PTX::UInt16Type>::value || std::is_same<T, PTX::Bit16Type>::value)
	{
		return LDInstruction::Type::U16;
	}
	else if constexpr(std::is_same<T, PTX::Int16Type>::value)
	{
		return LDInstruction::Type::S16;
	}
	else if constexpr(T::TypeBits == PTX::Bits::Bits32)
	{
		return LDInstruction::Type::X32;
	}
	else if constexpr(T::TypeBits == PTX::Bits::Bits64)
	{
		return LDInstruction::Type::X64;
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

	// Verify permissible synchronization

	if constexpr(A != PTX::LoadSynchronization::Weak)
	{
		Error(instruction, "unsupported synchronzation modifier");
	}
	else
	{
		ArchitectureDispatch::Dispatch(*this, instruction);
	}
}

template<PTX::Bits B, class T, class S, PTX::LoadSynchronization A>
void LoadGenerator::GenerateMaxwell(const PTX::LoadInstruction<B, T, S, A> *instruction)
{
	GenerateInstruction<
		SASS::Maxwell::MOVInstruction, SASS::Maxwell::LDGInstruction, SASS::Maxwell::LDSInstruction
	>(instruction);
}

template<PTX::Bits B, class T, class S, PTX::LoadSynchronization A>
void LoadGenerator::GenerateVolta(const PTX::LoadInstruction<B, T, S, A> *instruction)
{
	GenerateInstruction<
		SASS::Volta::MOVInstruction, SASS::Volta::LDGInstruction, SASS::Volta::LDSInstruction
	>(instruction);
}

template<class MOVInstruction, class LDGInstruction, class LDSInstruction, PTX::Bits B, class T, class S, PTX::LoadSynchronization A>
void LoadGenerator::GenerateInstruction(const PTX::LoadInstruction<B, T, S, A> *instruction)
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

		this->AddInstruction(new MOVInstruction(destination_Lo, address_Lo));

		// Extended datatypes

		if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			this->AddInstruction(new MOVInstruction(destination_Hi, address_Hi));
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

		// Generate instruction

		if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
		{
			// Flags

			auto type = InstructionType<LDGInstruction, T>();
			auto flags = LDGInstruction::Flags::None;
			if constexpr(B == PTX::Bits::Bits64)
			{
				flags |= LDGInstruction::Flags::E;
			}

			// Cache depends on the instruction

			if constexpr(std::is_same<LDGInstruction, SASS::Volta::LDGInstruction>::value)
			{
				auto cache = LDGInstruction::Cache::None;
				auto evict = LDGInstruction::Evict::None;
				auto prefetch = LDGInstruction::Prefetch::None;

				switch (instruction->GetCacheOperator())
				{
					case PTX::LoadInstruction<B, T, S, A>::CacheOperator::All:
					{
						cache = LDGInstruction::Cache::STRONG_SM;
						break;
					}
					case PTX::LoadInstruction<B, T, S, A>::CacheOperator::Global:
					{
						cache = LDGInstruction::Cache::STRONG_GPU;
						break;
					}
					case PTX::LoadInstruction<B, T, S, A>::CacheOperator::Streaming:
					{
						evict = LDGInstruction::Evict::EF;
						break;
					}
					case PTX::LoadInstruction<B, T, S, A>::CacheOperator::LastUse:
					{
						evict = LDGInstruction::Evict::LU;
						break;
					}
					case PTX::LoadInstruction<B, T, S, A>::CacheOperator::Invalidate:
					{
						cache = LDGInstruction::Cache::STRONG_SYS;
						break;
					}
				}

				this->AddInstruction(new LDGInstruction(destination, address, type, cache, evict, prefetch, flags));
			}
			else
			{
				auto cache = LDGInstruction::Cache::None;
				switch (instruction->GetCacheOperator())
				{
					// case PTX::LoadInstruction<B, T, S, A>::CacheOperator::All:
					// case PTX::LoadInstruction<B, T, S, A>::CacheOperator::Streaming:
					// {
					// 	cache = LDGInstruction::Cache::None;
					// 	break;
					// }
					case PTX::LoadInstruction<B, T, S, A>::CacheOperator::Global:
					case PTX::LoadInstruction<B, T, S, A>::CacheOperator::LastUse:
					{
						cache = LDGInstruction::Cache::CG;
						break;
					}
					case PTX::LoadInstruction<B, T, S, A>::CacheOperator::Invalidate:
					{
						cache = LDGInstruction::Cache::CV;
						break;
					}
				}

				this->AddInstruction(new LDGInstruction(destination, address, type, cache, flags));
			}
		}
		else if constexpr(std::is_same<S, PTX::SharedSpace>::value)
		{
			// Flags

			auto type = InstructionType<LDSInstruction, T>();

			// Flag not necessary for shared variables, removed in Volta
			//
			// auto flags = LDSInstruction::Flags::None;
			// if constexpr(B == PTX::Bits::Bits64)
			// {
			// 	flags |= LDSInstruction::Flags::E;
			// }

			this->AddInstruction(new LDSInstruction(destination, address, type));
		}
		else
		{
			Error(instruction, "unsupported space");
		}
	}
}

}
}
