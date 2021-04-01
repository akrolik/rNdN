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
	auto [destination, destination_Hi] = registerGenerator.Generate(instruction->GetDestination());
	auto [value, value_Hi] = registerGenerator.Generate(instruction->GetValue());

	// If CAS mode, the new value is sequential to the comparison value

	SASS::Register *sourceC = nullptr;
	SASS::Register *sourceC_Hi = nullptr;

	if constexpr(PTX::is_bit_type<T>::value)
	{
		if (instruction->GetOperation() == T::AtomicOperation::CompareAndSwap)
		{
			// The first temporary *must* be aligned to a multiple of (2xBitSize)

			auto align1 = Utils::Math::DivUp(PTX::BitSize<T::TypeBits>::NumBits * 2, 32);
			auto align2 = Utils::Math::DivUp(PTX::BitSize<T::TypeBits>::NumBits, 32);

			auto [temp0, temp1] = this->m_builder.AllocateTemporaryRegisterPair<T::TypeBits>(align1);
			auto [temp2, temp3] = this->m_builder.AllocateTemporaryRegisterPair<T::TypeBits>(align2);

			// Move values into temporaries
			
			this->AddInstruction(new SASS::MOVInstruction(temp0, value));
			if constexpr(T::TypeBits == PTX::Bits::Bits64)
			{
				this->AddInstruction(new SASS::MOVInstruction(temp1, value_Hi));
			}

			if (auto valueExtra = instruction->GetValueC())
			{
				CompositeGenerator compositeGenerator(this->m_builder);
				auto [valueC, valueC_Hi] = compositeGenerator.Generate(valueExtra);

				this->AddInstruction(new SASS::MOVInstruction(temp2, valueC));
				if constexpr(T::TypeBits == PTX::Bits::Bits64)
				{
					this->AddInstruction(new SASS::MOVInstruction(temp3, valueC_Hi));
				}
			}

			value = temp0;
			value_Hi = temp1;

			sourceC = temp2;
			sourceC_Hi = temp3;
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
