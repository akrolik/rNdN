#include "Backend/Codegen/Generators/Instructions/Synchronization/AtomicGenerator.h"

#include "Backend/Codegen/Generators/Operands/AddressGenerator.h"
#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void AtomicGenerator::Generate(const PTX::_AtomicInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<PTX::Bits B, class T, class S>
SASS::ATOMInstruction::Type AtomicGenerator::InstructionType(const PTX::AtomicInstruction<B, T, S> *instruction)
{
	if constexpr(std::is_same<T, PTX::UInt32Type>::value || std::is_same<T, PTX::Bit32Type>::value)
	{
		return SASS::ATOMInstruction::Type::U32;
	}
	else if constexpr(std::is_same<T, PTX::Int32Type>::value)
	{
		return SASS::ATOMInstruction::Type::S32;
	}
	else if constexpr(std::is_same<T, PTX::UInt64Type>::value)
	{
		return SASS::ATOMInstruction::Type::U64;
	}
	else if constexpr(std::is_same<T, PTX::Int64Type>::value)
	{
		return SASS::ATOMInstruction::Type::S64;
	}
	else if constexpr(std::is_same<T, PTX::Float16x2Type>::value)
	{
		return SASS::ATOMInstruction::Type::F16;
	}
	else if constexpr(std::is_same<T, PTX::Float32Type>::value)
	{
		return SASS::ATOMInstruction::Type::F32;
	}
	else if constexpr(std::is_same<T, PTX::Float64Type>::value || std::is_same<T, PTX::Bit64Type>::value)
	{
		return SASS::ATOMInstruction::Type::X64;
	}
	Error(instruction, "unsupported type");
}

template<PTX::Bits B, class T, class S>
SASS::ATOMInstruction::Mode AtomicGenerator::InstructionMode(const PTX::AtomicInstruction<B, T, S> *instruction)
{
	if constexpr(PTX::is_bit_type<T>::value)
	{
		switch (instruction->GetOperation())
		{
			case T::AtomicOperation::And:
				return SASS::ATOMInstruction::Mode::AND;
			case T::AtomicOperation::Or:
				return SASS::ATOMInstruction::Mode::OR;
			case T::AtomicOperation::Xor:
				return SASS::ATOMInstruction::Mode::XOR;
			case T::AtomicOperation::Exchange:
				return SASS::ATOMInstruction::Mode::EXCH;
			case T::AtomicOperation::CompareAndSwap:
				return SASS::ATOMInstruction::Mode::CAS;
		}
	}
	else if constexpr(std::is_same<T, PTX::Float16x2Type>::value)
	{
		if (instruction->GetOperation() == T::AtomicOperation::Add)
		{
			return SASS::ATOMInstruction::Mode::ADD;
		}
	}
	else
	{
		switch (instruction->GetOperation())
		{
			case T::AtomicOperation::Add:
				return SASS::ATOMInstruction::Mode::ADD;
			case T::AtomicOperation::Increment:
				return SASS::ATOMInstruction::Mode::INC;
			case T::AtomicOperation::Decrement:
				return SASS::ATOMInstruction::Mode::DEC;
			case T::AtomicOperation::Minimum:
				return SASS::ATOMInstruction::Mode::MIN;
			case T::AtomicOperation::Maximum:
				return SASS::ATOMInstruction::Mode::MAX;
		}
	}

	Error(instruction, "unsupported operation");
}

template<PTX::Bits B, class T, class S>
void AtomicGenerator::Visit(const PTX::AtomicInstruction<B, T, S> *instruction)
{
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

	// Generate operands

	RegisterGenerator registerGenerator(this->m_builder);
	auto destination = registerGenerator.Generate(instruction->GetDestination());
	auto value = registerGenerator.Generate(instruction->GetValue());

	// If CAS mode, the new value is sequential to the comparison value

	SASS::Register *sourceC = nullptr;

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
			
			this->AddInstruction(new SASS::MOVInstruction(temp0_Lo, value_Lo));
			if constexpr(T::TypeBits == PTX::Bits::Bits64)
			{
				this->AddInstruction(new SASS::MOVInstruction(temp0_Hi, value_Hi));
			}

			if (auto valueExtra = instruction->GetValueC())
			{
				CompositeGenerator compositeGenerator(this->m_builder);
				auto [valueC_Lo, valueC_Hi] = compositeGenerator.GeneratePair(valueExtra);

				this->AddInstruction(new SASS::MOVInstruction(temp1_Lo, valueC_Lo));
				if constexpr(T::TypeBits == PTX::Bits::Bits64)
				{
					this->AddInstruction(new SASS::MOVInstruction(temp1_Hi, valueC_Hi));
				}
			}

			// Assign temporaries for use with atomic operation

			auto size = Utils::Math::DivUp(PTX::BitSize<T::TypeBits>::NumBits, 32);

			value = new SASS::Register(temp0_Lo->GetValue(), size);
			sourceC = new SASS::Register(temp1_Lo->GetValue(), size);
		}
	}

	AddressGenerator addressGenerator(this->m_builder);
	auto address = addressGenerator.Generate(instruction->GetAddress());

	// Generate instruction

	if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
	{
		auto type = InstructionType(instruction);
		auto flags = SASS::ATOMInstruction::Flags::None;
		if constexpr(B == PTX::Bits::Bits64)
		{
			flags |= SASS::ATOMInstruction::Flags::E;
		}
		auto mode = InstructionMode(instruction);

		this->AddInstruction(new SASS::ATOMInstruction(destination, address, value, sourceC, type, mode, flags));
		this->AddInstruction(new SASS::DEPBARInstruction(
			SASS::DEPBARInstruction::Barrier::SB0, new SASS::I8Immediate(0x0), SASS::DEPBARInstruction::Flags::LE
		));
	}
	else
	{
		Error(instruction, "unsupported space");
	}
}

}
}
