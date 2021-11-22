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

template<class I, PTX::Bits B, class T, class S>
typename I::Type AtomicGenerator::InstructionType(const PTX::AtomicInstruction<B, T, S> *instruction)
{
	if constexpr(std::is_same<T, PTX::UInt32Type>::value || std::is_same<T, PTX::Bit32Type>::value)
	{
		return I::Type::U32;
	}
	else if constexpr(std::is_same<T, PTX::Int32Type>::value)
	{
		return I::Type::S32;
	}
	else if constexpr(std::is_same<T, PTX::UInt64Type>::value || std::is_same<T, PTX::Bit64Type>::value)
	{
		return I::Type::U64;
	}
	else if constexpr(std::is_same<T, PTX::Int64Type>::value)
	{
		return I::Type::S64;
	}
	else if constexpr(std::is_same<T, PTX::Float16x2Type>::value)
	{
		return I::Type::F16x2;
	}
	else if constexpr(std::is_same<T, PTX::Float32Type>::value)
	{
		return I::Type::F32;
	}
	else if constexpr(std::is_same<T, PTX::Float64Type>::value)
	{
		return I::Type::F64;
	}
	Error(instruction, "unsupported type");
}

template<class I, PTX::Bits B, class T, class S>
typename I::Mode AtomicGenerator::InstructionMode(const PTX::AtomicInstruction<B, T, S> *instruction)
{
	if constexpr(PTX::is_bit_type<T>::value)
	{
		switch (instruction->GetOperation())
		{
			case T::AtomicOperation::And:
			{
				return I::Mode::AND;
			}
			case T::AtomicOperation::Or:
			{
				return I::Mode::OR;
			}
			case T::AtomicOperation::Xor:
			{
				return I::Mode::XOR;
			}
			case T::AtomicOperation::Exchange:
			{
				return I::Mode::EXCH;
			}
		}
	}
	else if constexpr(std::is_same<T, PTX::Float16x2Type>::value)
	{
		if (instruction->GetOperation() == T::AtomicOperation::Add)
		{
			return I::Mode::ADD;
		}
	}
	else
	{
		switch (instruction->GetOperation())
		{
			case T::AtomicOperation::Add:
			{
				return I::Mode::ADD;
			}
			case T::AtomicOperation::Increment:
			{
				return I::Mode::INC;
			}
			case T::AtomicOperation::Decrement:
			{
				return I::Mode::DEC;
			}
			case T::AtomicOperation::Minimum:
			{
				return I::Mode::MIN;
			}
			case T::AtomicOperation::Maximum:
			{
				return I::Mode::MAX;
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

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<PTX::Bits B, class T, class S>
void AtomicGenerator::GenerateMaxwell(const PTX::AtomicInstruction<B, T, S> *instruction)
{
	// // Verify permissible properties

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
			
			this->AddInstruction(new SASS::Maxwell::MOVInstruction(temp0_Lo, value_Lo));
			if constexpr(T::TypeBits == PTX::Bits::Bits64)
			{
				this->AddInstruction(new SASS::Maxwell::MOVInstruction(temp0_Hi, value_Hi));
			}

			if (auto valueExtra = instruction->GetValueC())
			{
				CompositeGenerator compositeGenerator(this->m_builder);
				auto [valueC_Lo, valueC_Hi] = compositeGenerator.GeneratePair(valueExtra);

				this->AddInstruction(new SASS::Maxwell::MOVInstruction(temp1_Lo, valueC_Lo));
				if constexpr(T::TypeBits == PTX::Bits::Bits64)
				{
					this->AddInstruction(new SASS::Maxwell::MOVInstruction(temp1_Hi, valueC_Hi));
				}
			}

			// Assign temporaries for use with atomic operation

			auto size = Utils::Math::DivUp(PTX::BitSize<T::TypeBits>::NumBits, 32);

			auto sourceA = new SASS::Register(temp0_Lo->GetValue(), size);
			auto sourceB = new SASS::Register(temp1_Lo->GetValue(), size);

			// Generate instruction

			if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
			{
				auto type = SASS::Maxwell::ATOMCASInstruction::Type::X32;
				if constexpr(T::TypeBits == PTX::Bits::Bits64)
				{
					type = SASS::Maxwell::ATOMCASInstruction::Type::X64;
				}

				auto flags = SASS::Maxwell::ATOMCASInstruction::Flags::None;
				if constexpr(B == PTX::Bits::Bits64)
				{
					flags |= SASS::Maxwell::ATOMCASInstruction::Flags::E;
				}

				this->AddInstruction(new SASS::Maxwell::ATOMCASInstruction(destination, address, sourceA, sourceB, type, flags));
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
		auto type = InstructionType<SASS::Maxwell::ATOMInstruction>(instruction);
		auto flags = SASS::Maxwell::ATOMInstruction::Flags::None;
		if constexpr(B == PTX::Bits::Bits64)
		{
			flags |= SASS::Maxwell::ATOMInstruction::Flags::E;
		}
		auto mode = InstructionMode<SASS::Maxwell::ATOMInstruction>(instruction);

		this->AddInstruction(new SASS::Maxwell::ATOMInstruction(destination, address, value, type, mode, flags));
	}
	else
	{
		Error(instruction, "unsupported space");
	}
}

template<PTX::Bits B, class T, class S>
void AtomicGenerator::GenerateVolta(const PTX::AtomicInstruction<B, T, S> *instruction)
{
	Error(instruction, "unsupported architecture");
}

}
}
