#include "Backend/Codegen/Generators/Instructions/Synchronization/ReductionGenerator.h"

#include "Backend/Codegen/Generators/Operands/AddressGenerator.h"
#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void ReductionGenerator::Generate(const PTX::_ReductionInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class I, PTX::Bits B, class T, class S>
typename I::Type ReductionGenerator::InstructionType(const PTX::ReductionInstruction<B, T, S> *instruction)
{
	if constexpr(std::is_same<T, PTX::UInt32Type>::value)
	{
		return I::Type::U32;
	}
	else if constexpr(std::is_same<T, PTX::Int32Type>::value)
	{
		return I::Type::S32;
	}
	else if constexpr(std::is_same<T, PTX::UInt64Type>::value)
	{
		return I::Type::U64;
	}
	else if constexpr(std::is_same<T, PTX::Int64Type>::value)
	{
		return I::Type::S64;
	}
	else if constexpr(std::is_same<T, PTX::Float16x2Type>::value)
	{
		return I::Type::F16;
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
typename I::Mode ReductionGenerator::InstructionMode(const PTX::ReductionInstruction<B, T, S> *instruction)
{
	if constexpr(PTX::is_bit_type<T>::value)
	{
		switch (instruction->GetOperation())
		{
			case T::ReductionOperation::And:
			{
				return I::Mode::AND;
			}
			case T::ReductionOperation::Or:
			{
				return I::Mode::OR;
			}
			case T::ReductionOperation::Xor:
			{
				return I::Mode::XOR;
			}
		}
	}
	else if constexpr(std::is_same<T, PTX::Float16x2Type>::value)
	{
		if (instruction->GetOperation() == T::ReductionOperation::Add)
		{
			return I::Mode::ADD;
		}
	}
	else
	{
		switch (instruction->GetOperation())
		{
			case T::ReductionOperation::Add:
			{
				return I::Mode::ADD;
			}
			case T::ReductionOperation::Increment:
			{
				return I::Mode::INC;
			}
			case T::ReductionOperation::Decrement:
			{
				return I::Mode::DEC;
			}
			case T::ReductionOperation::Minimum:
			{
				return I::Mode::MIN;
			}
			case T::ReductionOperation::Maximum:
			{
				return I::Mode::MAX;
			}
		}
	}
	Error(instruction, "unsupported operation");
}

template<PTX::Bits B, class T, class S>
void ReductionGenerator::Visit(const PTX::ReductionInstruction<B, T, S> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

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

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<PTX::Bits B, class T, class S>
void ReductionGenerator::GenerateMaxwell(const PTX::ReductionInstruction<B, T, S> *instruction)
{
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
	auto value = registerGenerator.Generate(instruction->GetValue());

	// Generate instruction

	if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
	{
		auto type = InstructionType<SASS::Maxwell::REDInstruction>(instruction);
		auto flags = SASS::Maxwell::REDInstruction::Flags::None;
		if constexpr(B == PTX::Bits::Bits64)
		{
			flags |= SASS::Maxwell::REDInstruction::Flags::E;
		}
		auto mode = InstructionMode<SASS::Maxwell::REDInstruction>(instruction);

		this->AddInstruction(new SASS::Maxwell::REDInstruction(address, value, type, mode, flags));
	}
}

template<PTX::Bits B, class T, class S>
void ReductionGenerator::GenerateVolta(const PTX::ReductionInstruction<B, T, S> *instruction)
{
	Error(instruction, "unsupported architecture");
}

}
}
