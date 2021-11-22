#include "Backend/Codegen/Generators/Instructions/Arithmetic/AddGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void AddGenerator::Generate(const PTX::_AddInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void AddGenerator::Visit(const PTX::AddInstruction<T> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types:
	//   - Int16, Int32, Int64
	//   - UInt16, UInt32, UInt64
	//   - Float16, Float16x2, Float32, Float64
	// Modifiers:
	//   - Carry: Int32, Int64, UInt32, UInt64
	//   - FlushSubnormal: Float16, Float16x2, Float32
	//   - Rounding: Float16, Float16x2, Float32, Float64
	//   - Saturate: Int32, Float16, Float16x2, Float32

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<class T>
void AddGenerator::GenerateMaxwell(const PTX::AddInstruction<T> *instruction)
{
	RegisterGenerator registerGenerator(this->m_builder);
	CompositeGenerator compositeGenerator(this->m_builder);

	if constexpr(PTX::is_int_type<T>::value)
	{
		auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());
		auto [sourceA_Lo, sourceA_Hi] = registerGenerator.GeneratePair(instruction->GetSourceA());
		auto [sourceB_Lo, sourceB_Hi] = compositeGenerator.GeneratePair(instruction->GetSourceB());

		// Carry modifier

		if constexpr(T::TypeBits == PTX::Bits::Bits16 || T::TypeBits == PTX::Bits::Bits32)
		{
			// Carry flags

			auto flags = SASS::Maxwell::IADDInstruction::Flags::None;
			if constexpr(T::TypeBits == PTX::Bits::Bits32)
			{
				if (instruction->GetCarryIn())
				{
					flags |= SASS::Maxwell::IADDInstruction::Flags::X;
				}
				if (instruction->GetCarryOut())
				{
					flags |= SASS::Maxwell::IADDInstruction::Flags::CC;
				}
			}

			// Saturate modifier

			if constexpr(std::is_same<T, PTX::Int32Type>::value)
			{
				if (!instruction->PTX::CarryModifier<T>::IsActive() && instruction->GetSaturate())
				{
					flags |= SASS::Maxwell::IADDInstruction::Flags::SAT;
				}
			}

			this->AddInstruction(new SASS::Maxwell::IADDInstruction(destination_Lo, sourceA_Lo, sourceB_Lo, flags));

			// Keep in range for 16-bit

			if constexpr(std::is_same<T, PTX::UInt16Type>::value)
			{
				this->AddInstruction(new SASS::Maxwell::LOP32IInstruction(
					destination_Lo, destination_Lo, new SASS::I32Immediate(0xffff), SASS::Maxwell::LOP32IInstruction::BooleanOperator::AND
				));
			}
			else if constexpr(std::is_same<T, PTX::Int16Type>::value)
			{
				this->AddInstruction(new SASS::Maxwell::BFEInstruction(
					destination_Lo, destination_Lo, new SASS::I32Immediate(0x1000))
				);
			}
		}
		else if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			// Carry flags

			auto flags1 = SASS::Maxwell::IADDInstruction::Flags::None;
			auto flags2 = SASS::Maxwell::IADDInstruction::Flags::None;

			if (instruction->GetCarryIn())
			{
				flags1 |= SASS::Maxwell::IADDInstruction::Flags::X;
			}
			if (instruction->GetCarryOut())
			{
				flags2 |= SASS::Maxwell::IADDInstruction::Flags::CC;
			}

			// Extended add

			this->AddInstruction(new SASS::Maxwell::IADDInstruction(
				destination_Lo, sourceA_Lo, sourceB_Lo, flags1 | SASS::Maxwell::IADDInstruction::Flags::CC
			));
			this->AddInstruction(new SASS::Maxwell::IADDInstruction(
				destination_Hi, sourceA_Hi, sourceB_Hi, flags2 | SASS::Maxwell::IADDInstruction::Flags::X
			));
		}
	}
	else if constexpr(std::is_same<T, PTX::Float64Type>::value)
	{
		// Generate operands

		auto destination = registerGenerator.Generate(instruction->GetDestination());
		auto sourceA = registerGenerator.Generate(instruction->GetSourceA());
		auto sourceB = compositeGenerator.Generate(instruction->GetSourceB());

		// Generate instruction

		auto round = SASS::Maxwell::DADDInstruction::Round::RN;
		switch (instruction->GetRoundingMode())
		{
			// case T::RoundingMode::None:
			// case T::RoundingMode::Nearest:
			// {
			// 	round = SASS::Maxwell::DADDInstruction::Round::RN;
			// 	break;
			// }
			case T::RoundingMode::Zero:
			{
				round = SASS::Maxwell::DADDInstruction::Round::RZ;
				break;
			}
			case T::RoundingMode::NegativeInfinity:
			{
				round = SASS::Maxwell::DADDInstruction::Round::RM;
				break;
			}
			case T::RoundingMode::PositiveInfinity:
			{
				round = SASS::Maxwell::DADDInstruction::Round::RP;
				break;
			}
		}

		this->AddInstruction(new SASS::Maxwell::DADDInstruction(destination, sourceA, sourceB, round));
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

template<class T>
void AddGenerator::GenerateVolta(const PTX::AddInstruction<T> *instruction)
{
	Error(instruction, "unsupported architecture");
}

}
}
