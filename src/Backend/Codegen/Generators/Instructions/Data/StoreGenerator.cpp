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

template<typename STInstruction, class T>
typename STInstruction::Type StoreGenerator::InstructionType()
{
	if constexpr(std::is_same<T, PTX::UInt8Type>::value || std::is_same<T, PTX::Bit8Type>::value)
	{
		return STInstruction::Type::U8;
	}
	else if constexpr(std::is_same<T, PTX::Int8Type>::value)
	{
		return STInstruction::Type::S8;
	}
	else if constexpr(std::is_same<T, PTX::UInt16Type>::value || std::is_same<T, PTX::Bit16Type>::value)
	{
		return STInstruction::Type::U16;
	}
	else if constexpr(std::is_same<T, PTX::Int16Type>::value)
	{
		return STInstruction::Type::S16;
	}
	else if constexpr(T::TypeBits == PTX::Bits::Bits32)
	{
		return STInstruction::Type::X32;
	}
	else if constexpr(T::TypeBits == PTX::Bits::Bits64)
	{
		return STInstruction::Type::X64;
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

	// Verify permissible synchronization

	if constexpr(A != PTX::StoreSynchronization::Weak)
	{
		Error(instruction, "unsupported synchronzation modifier");
	}
	else
	{
		ArchitectureDispatch::Dispatch(*this, instruction);
	}
}

template<PTX::Bits B, class T, class S, PTX::StoreSynchronization A>
void StoreGenerator::GenerateMaxwell(const PTX::StoreInstruction<B, T, S, A> *instruction)
{
	GenerateInstruction<
		SASS::Maxwell::STGInstruction, SASS::Maxwell::STSInstruction
	>(instruction);
}

template<PTX::Bits B, class T, class S, PTX::StoreSynchronization A>
void StoreGenerator::GenerateVolta(const PTX::StoreInstruction<B, T, S, A> *instruction)
{
	GenerateInstruction<
		SASS::Volta::STGInstruction, SASS::Volta::STSInstruction
	>(instruction);
}

template<class STGInstruction, class STSInstruction, PTX::Bits B, class T, class S, PTX::StoreSynchronization A>
void StoreGenerator::GenerateInstruction(const PTX::StoreInstruction<B, T, S, A> *instruction)
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

		auto type = InstructionType<STGInstruction, T>();
		auto flags = STGInstruction::Flags::None;
		if constexpr(B == PTX::Bits::Bits64)
		{
			flags = STGInstruction::Flags::E;
		}

		// Cache mode depends on architecture

		if constexpr(std::is_same<STGInstruction, SASS::Volta::STGInstruction>::value)
		{
			auto cache = STGInstruction::Cache::None;
			auto evict = STGInstruction::Evict::None;

			switch (instruction->GetCacheOperator())
			{
				case PTX::StoreInstruction<B, T, S, A>::CacheOperator::WriteBack:
				{
					cache = STGInstruction::Cache::STRONG_SYS;
					break;
				}
				case PTX::StoreInstruction<B, T, S, A>::CacheOperator::Global:
				{
					cache = STGInstruction::Cache::STRONG_GPU;
					break;
				}
				case PTX::StoreInstruction<B, T, S, A>::CacheOperator::Streaming:
				{
					evict = STGInstruction::Evict::EF;
					break;
				}
				case PTX::StoreInstruction<B, T, S, A>::CacheOperator::WriteThrough:
				{
					cache = STGInstruction::Cache::STRONG_SYS;
					break;
				}
			}

			this->AddInstruction(new STGInstruction(address, source, type, cache, evict, flags));
		}
		else
		{
			auto cache = STGInstruction::Cache::None;
			switch (instruction->GetCacheOperator())
			{
				// case PTX::StoreInstruction<B, T, S, A>::CacheOperator::WriteBack:
				// {
				// 	cache = STGInstruction::Cache::None;
				// 	break;
				// }
				case PTX::StoreInstruction<B, T, S, A>::CacheOperator::Global:
				{
					cache = STGInstruction::Cache::CG;
					break;
				}
				case PTX::StoreInstruction<B, T, S, A>::CacheOperator::Streaming:
				{
					cache = STGInstruction::Cache::CS;
					break;
				}
				case PTX::StoreInstruction<B, T, S, A>::CacheOperator::WriteThrough:
				{
					cache = STGInstruction::Cache::WT;
					break;
				}
			}

			this->AddInstruction(new STGInstruction(address, source, type, cache, flags));
		}
	}
	else if constexpr(std::is_same<S, PTX::SharedSpace>::value)
	{
		// Generate instruction

		auto type = InstructionType<STSInstruction, T>();

		// Flag not necessary for shared variables, removed in Volta
		//
		// auto flags = STSInstruction::Flags::None;
		// if constexpr(B == PTX::Bits::Bits64)
		// {
		// 	flags |= STSInstruction::Flags::E;
		// }

		this->AddInstruction(new STSInstruction(address, source, type));
	}
	else
	{
		Error(instruction, "unsupported space");
	}
}

}
}
