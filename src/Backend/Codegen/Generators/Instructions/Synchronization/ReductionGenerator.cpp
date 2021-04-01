#include "Backend/Codegen/Generators/Instructions/Synchronization/ReductionGenerator.h"

#include "Backend/Codegen/Generators/Operands/AddressGenerator.h"
#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void ReductionGenerator::Generate(const PTX::_ReductionInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<PTX::Bits B, class T, class S>
SASS::REDInstruction::Type ReductionGenerator::InstructionType(const PTX::ReductionInstruction<B, T, S> *instruction)
{
	if constexpr(std::is_same<T, PTX::UInt32Type>::value)
	{
		return SASS::REDInstruction::Type::U32;
	}
	else if constexpr(std::is_same<T, PTX::Int32Type>::value)
	{
		return SASS::REDInstruction::Type::S32;
	}
	else if constexpr(std::is_same<T, PTX::UInt64Type>::value)
	{
		return SASS::REDInstruction::Type::U64;
	}
	else if constexpr(std::is_same<T, PTX::Int64Type>::value)
	{
		return SASS::REDInstruction::Type::S64;
	}
	else if constexpr(std::is_same<T, PTX::Float16x2Type>::value)
	{
		return SASS::REDInstruction::Type::F16;
	}
	else if constexpr(std::is_same<T, PTX::Float32Type>::value)
	{
		return SASS::REDInstruction::Type::F32;
	}
	else if constexpr(std::is_same<T, PTX::Float64Type>::value)
	{
		return SASS::REDInstruction::Type::F64;
	}
	Error(instruction, "unsupported type");
}

template<PTX::Bits B, class T, class S>
SASS::REDInstruction::Mode ReductionGenerator::InstructionMode(const PTX::ReductionInstruction<B, T, S> *instruction)
{
	if constexpr(PTX::is_bit_type<T>::value)
	{
		switch (instruction->GetOperation())
		{
			case T::ReductionOperation::And:
				return SASS::REDInstruction::Mode::AND;
			case T::ReductionOperation::Or:
				return SASS::REDInstruction::Mode::OR;
			case T::ReductionOperation::Xor:
				return SASS::REDInstruction::Mode::XOR;
		}
	}
	else if constexpr(std::is_same<T, PTX::Float16x2Type>::value)
	{
		if (instruction->GetOperation() == T::ReductionOperation::Add)
		{
			return SASS::REDInstruction::Mode::ADD;
		}
	}
	else
	{
		switch (instruction->GetOperation())
		{
			case T::ReductionOperation::Add:
				return SASS::REDInstruction::Mode::ADD;
			case T::ReductionOperation::Increment:
				return SASS::REDInstruction::Mode::INC;
			case T::ReductionOperation::Decrement:
				return SASS::REDInstruction::Mode::DEC;
			case T::ReductionOperation::Minimum:
				return SASS::REDInstruction::Mode::MIN;
			case T::ReductionOperation::Maximum:
				return SASS::REDInstruction::Mode::MAX;
		}
	}
	Error(instruction, "unsupported operation");
}

template<PTX::Bits B, class T, class S>
void ReductionGenerator::Visit(const PTX::ReductionInstruction<B, T, S> *instruction)
{
	// Types
	//   - Bit32, Bit64
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
	if (synchronization != PTX::ReductionInstruction<B, T, S>::Synchronization::None)
	{
		Error(instruction, "unsupported synchronization modifier");
	}
	auto scope = instruction->GetScope(); 
	if (scope != PTX::ReductionInstruction<B, T, S>::Scope::None)
	{
		Error(instruction, "unsupported scope modifier");
	}

	// Generate operands

	AddressGenerator addressGenerator(this->m_builder);
	auto address = addressGenerator.Generate(instruction->GetAddress());

	RegisterGenerator registerGenerator(this->m_builder);
	auto [value, value_Hi] = registerGenerator.Generate(instruction->GetValue());

	// Generate instruction

	if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
	{
		auto type = InstructionType(instruction);
		auto flags = SASS::REDInstruction::Flags::None;
		if constexpr(B == PTX::Bits::Bits64)
		{
			flags |= SASS::REDInstruction::Flags::E;
		}
		auto mode = InstructionMode(instruction);

		this->AddInstruction(new SASS::REDInstruction(address, value, type, mode, flags));
		this->AddInstruction(new SASS::DEPBARInstruction(
			SASS::DEPBARInstruction::Barrier::SB0, new SASS::I8Immediate(0x0), SASS::DEPBARInstruction::Flags::LE
		));
	}
}

}
}
