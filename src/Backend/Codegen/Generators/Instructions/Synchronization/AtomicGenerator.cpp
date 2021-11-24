#include "Backend/Codegen/Generators/Instructions/Synchronization/AtomicGenerator.h"

#include "Backend/Codegen/Generators/Operands/AddressGenerator.h"
#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void AtomicGenerator::Generate(const PTX::_AtomicInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class ATOMInstruction, PTX::Bits B, class T, class S>
typename ATOMInstruction::Type AtomicGenerator::InstructionType(const PTX::AtomicInstruction<B, T, S> *instruction)
{
	if constexpr(std::is_same<T, PTX::UInt32Type>::value || std::is_same<T, PTX::Bit32Type>::value)
	{
		return ATOMInstruction::Type::U32;
	}
	else if constexpr(std::is_same<T, PTX::Int32Type>::value)
	{
		return ATOMInstruction::Type::S32;
	}
	else if constexpr(std::is_same<T, PTX::UInt64Type>::value || std::is_same<T, PTX::Bit64Type>::value)
	{
		return ATOMInstruction::Type::U64;
	}
	else if constexpr(std::is_same<T, PTX::Int64Type>::value)
	{
		return ATOMInstruction::Type::S64;
	}
	else if constexpr(std::is_same<T, PTX::Float16x2Type>::value)
	{
		return ATOMInstruction::Type::F16x2;
	}
	else if constexpr(std::is_same<T, PTX::Float32Type>::value)
	{
		return ATOMInstruction::Type::F32;
	}
	else if constexpr(std::is_same<T, PTX::Float64Type>::value)
	{
		return ATOMInstruction::Type::F64;
	}
	Error(instruction, "unsupported type");
}

template<class ATOMInstruction, PTX::Bits B, class T, class S>
typename ATOMInstruction::Mode AtomicGenerator::InstructionMode(const PTX::AtomicInstruction<B, T, S> *instruction)
{
	if constexpr(PTX::is_bit_type<T>::value)
	{
		switch (instruction->GetOperation())
		{
			case T::AtomicOperation::And:
			{
				return ATOMInstruction::Mode::AND;
			}
			case T::AtomicOperation::Or:
			{
				return ATOMInstruction::Mode::OR;
			}
			case T::AtomicOperation::Xor:
			{
				return ATOMInstruction::Mode::XOR;
			}
			case T::AtomicOperation::Exchange:
			{
				return ATOMInstruction::Mode::EXCH;
			}
		}
	}
	else if constexpr(std::is_same<T, PTX::Float16x2Type>::value)
	{
		if (instruction->GetOperation() == T::AtomicOperation::Add)
		{
			return ATOMInstruction::Mode::ADD;
		}
	}
	else
	{
		switch (instruction->GetOperation())
		{
			case T::AtomicOperation::Add:
			{
				return ATOMInstruction::Mode::ADD;
			}
			case T::AtomicOperation::Increment:
			{
				return ATOMInstruction::Mode::INC;
			}
			case T::AtomicOperation::Decrement:
			{
				return ATOMInstruction::Mode::DEC;
			}
			case T::AtomicOperation::Minimum:
			{
				return ATOMInstruction::Mode::MIN;
			}
			case T::AtomicOperation::Maximum:
			{
				return ATOMInstruction::Mode::MAX;
			}
		}
	}

	Error(instruction, "unsupported operation");
}

template<PTX::Bits B, class T, class S>
void AtomicGenerator::Visit(const PTX::AtomicInstruction<B, T, S> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types
	//   - Bit16, Bit32, Bit64
	//   - Int32, Int64
	//   - UInt32, UInt64
	//   - Float16, Float16x2, Float32, Float64
	// Spaces
	//   - AddressableSpace
	//   - GlobalSpace
	//   - SharedSpace
	// Modifiers
	//   - Scope: *

	// Verify permissible properties

	auto synchronization = instruction->GetSynchronization();
	if (synchronization != PTX::AtomicInstruction<B, T, S>::Synchronization::None)
	{
		Error(instruction, "unsupported synchronization modifier");
	}
	auto scope = instruction->GetScope(); 
	if (scope != PTX::AtomicInstruction<B, T, S>::Scope::None)
	{
		Error(instruction, "unsupported scope modifier");
	}

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<PTX::Bits B, class T, class S>
void AtomicGenerator::GenerateMaxwell(const PTX::AtomicInstruction<B, T, S> *instruction)
{
	GenerateInstruction<
		SASS::Maxwell::ATOMInstruction,
		SASS::Maxwell::ATOMCASInstruction,
		SASS::Maxwell::MOVInstruction
	>(instruction);
}

template<PTX::Bits B, class T, class S>
void AtomicGenerator::GenerateVolta(const PTX::AtomicInstruction<B, T, S> *instruction)
{
	GenerateInstruction<
		SASS::Volta::ATOMGInstruction,
		SASS::Volta::ATOMGInstruction,
		SASS::Volta::MOVInstruction
	>(instruction);
}

template<class ATOMInstruction, class ATOMCASInstruction, class MOVInstruction, PTX::Bits B, class T, class S>
void AtomicGenerator::GenerateInstruction(const PTX::AtomicInstruction<B, T, S> *instruction)
{
	// Generate operands

	RegisterGenerator registerGenerator(this->m_builder);
	auto destination = registerGenerator.Generate(instruction->GetDestination());
	auto value = registerGenerator.Generate(instruction->GetValue());

	AddressGenerator addressGenerator(this->m_builder);
	auto address = addressGenerator.Generate(instruction->GetAddress());

	// If CAS mode, the new value is sequential to the comparison value, and the instruction requires a special opcode

	if constexpr(PTX::is_bit_type<T>::value)
	{
		if (instruction->GetOperation() == T::AtomicOperation::CompareAndSwap)
		{
			// The first temporary *must* be aligned to a multiple of (2xBitSize)

			auto align1 = Utils::Math::DivUp(PTX::BitSize<T::TypeBits>::NumBits * 2, 32);
			auto align2 = Utils::Math::DivUp(PTX::BitSize<T::TypeBits>::NumBits, 32);

			auto [temp0_Lo, temp0_Hi] = this->m_builder.AllocateTemporaryRegisterPair<T::TypeBits>(align1);
			auto [temp1_Lo, temp1_Hi] = this->m_builder.AllocateTemporaryRegisterPair<T::TypeBits>(align2);

			// Move values into temporaries

			auto [value_Lo, value_Hi] = registerGenerator.GeneratePair(instruction->GetValue());
			
			this->AddInstruction(new MOVInstruction(temp0_Lo, value_Lo));
			if constexpr(T::TypeBits == PTX::Bits::Bits64)
			{
				this->AddInstruction(new MOVInstruction(temp0_Hi, value_Hi));
			}

			if (auto valueExtra = instruction->GetValueC())
			{
				CompositeGenerator compositeGenerator(this->m_builder);
				auto [valueC_Lo, valueC_Hi] = compositeGenerator.GeneratePair(valueExtra);

				this->AddInstruction(new MOVInstruction(temp1_Lo, valueC_Lo));
				if constexpr(T::TypeBits == PTX::Bits::Bits64)
				{
					this->AddInstruction(new MOVInstruction(temp1_Hi, valueC_Hi));
				}
			}

			// Assign temporaries for use with atomic operation

			auto size = Utils::Math::DivUp(PTX::BitSize<T::TypeBits>::NumBits, 32);

			auto sourceA = new SASS::Register(temp0_Lo->GetValue(), size);
			auto sourceB = new SASS::Register(temp1_Lo->GetValue(), size);

			// Generate instruction

			if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
			{
				auto flags = ATOMCASInstruction::Flags::None;
				if constexpr(B == PTX::Bits::Bits64)
				{
					flags |= ATOMCASInstruction::Flags::E;
				}

				if constexpr(std::is_same<ATOMCASInstruction, SASS::Volta::ATOMGInstruction>::value)
				{
					auto mode = ATOMCASInstruction::Mode::CAS;
					auto cache = ATOMCASInstruction::Cache::STRONG_GPU;
					auto type = ATOMCASInstruction::Type::U32;
					if constexpr(T::TypeBits == PTX::Bits::Bits64)
					{
						type = ATOMCASInstruction::Type::U64;
					}

					this->AddInstruction(new ATOMCASInstruction(SASS::PT, destination, address, sourceA, sourceB, type, mode, cache, flags));
				}
				else
				{
					auto type = ATOMCASInstruction::Type::X32;
					if constexpr(T::TypeBits == PTX::Bits::Bits64)
					{
						type = ATOMCASInstruction::Type::X64;
					}

					this->AddInstruction(new ATOMCASInstruction(destination, address, sourceA, sourceB, type, flags));
				}
				return;
			}
			else
			{
				Error(instruction, "unsupported space");
			}
		}
	}

	// Generate instruction

	if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
	{
		auto type = InstructionType<ATOMInstruction>(instruction);
		auto mode = InstructionMode<ATOMInstruction>(instruction);

		auto flags = ATOMInstruction::Flags::None;
		if constexpr(B == PTX::Bits::Bits64)
		{
			flags |= ATOMInstruction::Flags::E;
		}

		if constexpr(std::is_same<ATOMInstruction, SASS::Volta::ATOMGInstruction>::value)
		{
			auto cache = ATOMInstruction::Cache::STRONG_GPU;

			this->AddInstruction(new ATOMInstruction(SASS::PT, destination, address, value, nullptr, type, mode, cache, flags));
		}
		else
		{
			this->AddInstruction(new ATOMInstruction(destination, address, value, type, mode, flags));
		}
	}
	else
	{
		Error(instruction, "unsupported space");
	}
}

}
}
