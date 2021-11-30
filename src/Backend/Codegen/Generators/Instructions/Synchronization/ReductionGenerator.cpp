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

template<class REDInstruction, PTX::Bits B, class T, class S>
typename REDInstruction::Type ReductionGenerator::InstructionType(const PTX::ReductionInstruction<B, T, S> *instruction)
{
	if constexpr(std::is_same<T, PTX::UInt32Type>::value)
	{
		return REDInstruction::Type::U32;
	}
	else if constexpr(std::is_same<T, PTX::Int32Type>::value)
	{
		return REDInstruction::Type::S32;
	}
	else if constexpr(std::is_same<T, PTX::UInt64Type>::value)
	{
		return REDInstruction::Type::U64;
	}
	else if constexpr(std::is_same<T, PTX::Int64Type>::value)
	{
		return REDInstruction::Type::S64;
	}
	else if constexpr(std::is_same<T, PTX::Float16x2Type>::value)
	{
		return REDInstruction::Type::F16;
	}
	else if constexpr(std::is_same<T, PTX::Float32Type>::value)
	{
		return REDInstruction::Type::F32;
	}
	else if constexpr(std::is_same<T, PTX::Float64Type>::value)
	{
		return REDInstruction::Type::F64;
	}
	Error(instruction, "unsupported type");
}

template<class REDInstruction, PTX::Bits B, class T, class S>
typename REDInstruction::Mode ReductionGenerator::InstructionMode(const PTX::ReductionInstruction<B, T, S> *instruction)
{
	if constexpr(PTX::is_bit_type<T>::value)
	{
		switch (instruction->GetOperation())
		{
			case T::ReductionOperation::And:
			{
				return REDInstruction::Mode::AND;
			}
			case T::ReductionOperation::Or:
			{
				return REDInstruction::Mode::OR;
			}
			case T::ReductionOperation::Xor:
			{
				return REDInstruction::Mode::XOR;
			}
		}
	}
	else if constexpr(std::is_same<T, PTX::Float16x2Type>::value)
	{
		if (instruction->GetOperation() == T::ReductionOperation::Add)
		{
			return REDInstruction::Mode::ADD;
		}
	}
	else
	{
		switch (instruction->GetOperation())
		{
			case T::ReductionOperation::Add:
			{
				return REDInstruction::Mode::ADD;
			}
			case T::ReductionOperation::Increment:
			{
				return REDInstruction::Mode::INC;
			}
			case T::ReductionOperation::Decrement:
			{
				return REDInstruction::Mode::DEC;
			}
			case T::ReductionOperation::Minimum:
			{
				return REDInstruction::Mode::MIN;
			}
			case T::ReductionOperation::Maximum:
			{
				return REDInstruction::Mode::MAX;
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

	ArchitectureDispatch::DispatchInstruction<
		SASS::Maxwell::REDInstruction, SASS::Volta::REDInstruction
	>(*this, instruction);
}

template<class REDInstruction, PTX::Bits B, class T, class S>
void ReductionGenerator::GenerateInstruction(const PTX::ReductionInstruction<B, T, S> *instruction)
{
	// Generate operands

	AddressGenerator addressGenerator(this->m_builder);
	auto address = addressGenerator.Generate(instruction->GetAddress());

	RegisterGenerator registerGenerator(this->m_builder);
	auto value = registerGenerator.Generate(instruction->GetValue());

	// Generate instruction

	if constexpr(std::is_same<S, PTX::GlobalSpace>::value)
	{
		// Flags

		auto type = InstructionType<REDInstruction>(instruction);
		auto mode = InstructionMode<REDInstruction>(instruction);

		auto flags = REDInstruction::Flags::None;
		if constexpr(B == PTX::Bits::Bits64)
		{
			flags |= REDInstruction::Flags::E;
		}

		if constexpr(std::is_same<REDInstruction, SASS::Volta::REDInstruction>::value)
		{
			// Volta instruction requires cache type

			auto cache = REDInstruction::Cache::None;

			this->AddInstruction(new REDInstruction(address, value, type, mode, cache, flags));
		}
		else
		{
			this->AddInstruction(new REDInstruction(address, value, type, mode, flags));
		}
	}
	else
	{
		Error(instruction, "unsupported space");
	}
}

}
}
