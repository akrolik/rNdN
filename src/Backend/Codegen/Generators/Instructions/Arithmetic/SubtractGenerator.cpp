#include "Backend/Codegen/Generators/Instructions/Arithmetic/SubtractGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void SubtractGenerator::Generate(const PTX::_SubtractInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void SubtractGenerator::Visit(const PTX::SubtractInstruction<T> *instruction)
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

	// Generate operands

	RegisterGenerator registerGenerator(this->m_builder);
	CompositeGenerator compositeGenerator(this->m_builder);

	// Generate instruction

	if constexpr(PTX::is_int_type<T>::value)
	{
		auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());
		auto [sourceA_Lo, sourceA_Hi] = registerGenerator.GeneratePair(instruction->GetSourceA());
		auto [sourceB_Lo, sourceB_Hi] = compositeGenerator.GeneratePair(instruction->GetSourceB());

		auto flags1 = SASS::IADDInstruction::Flags::NEG_B;
		auto flags2 = SASS::IADDInstruction::Flags::NEG_B;

		// Carry modifier

		if constexpr(T::TypeBits == PTX::Bits::Bits32 || T::TypeBits == PTX::Bits::Bits64)
		{
			if (instruction->GetCarryIn())
			{
				flags1 |= SASS::IADDInstruction::Flags::X;
			}
			if (instruction->GetCarryOut())
			{
				flags2 |= SASS::IADDInstruction::Flags::CC;
			}
		}

		// Saturate modifier

		if constexpr(T::TypeBits == PTX::Bits::Bits16 || T::TypeBits == PTX::Bits::Bits32)
		{
			if constexpr(std::is_same<T, PTX::Int32Type>::value)
			{
				if (!instruction->PTX::CarryModifier<T>::IsActive() && instruction->GetSaturate())
				{
					flags1 |= SASS::IADDInstruction::Flags::SAT;
				}
			}

			this->AddInstruction(new SASS::IADDInstruction(destination_Lo, sourceA_Lo, sourceB_Lo, flags1));
		}
		else if constexpr(T::TypeBits == PTX::Bits::Bits64)
		{
			this->AddInstruction(new SASS::IADDInstruction(
				destination_Lo, sourceA_Lo, sourceB_Lo, flags1 | SASS::IADDInstruction::Flags::CC
			));
			this->AddInstruction(new SASS::IADDInstruction(
				destination_Hi, sourceA_Hi, sourceB_Hi, flags2 | SASS::IADDInstruction::Flags::X
			));
		}
	}
	else if constexpr(std::is_same<T, PTX::Float64Type>::value)
	{
		auto destination = registerGenerator.Generate(instruction->GetDestination());
		auto sourceA = registerGenerator.Generate(instruction->GetSourceA());
		auto sourceB = compositeGenerator.Generate(instruction->GetSourceB());

		auto round = SASS::DADDInstruction::Round::RN;
		switch (instruction->GetRoundingMode())
		{
			// case T::RoundingMode::None:
			// case T::RoundingMode::Nearest:
			// {
			// 	round = SASS::DADDInstruction::Round::RN;
			// 	break;
			// }
			case T::RoundingMode::Zero:
			{
				round = SASS::DADDInstruction::Round::RZ;
				break;
			}
			case T::RoundingMode::NegativeInfinity:
			{
				round = SASS::DADDInstruction::Round::RM;
				break;
			}
			case T::RoundingMode::PositiveInfinity:
			{
				round = SASS::DADDInstruction::Round::RP;
				break;
			}
		}

		this->AddInstruction(new SASS::DADDInstruction(destination, sourceA, sourceB, round, SASS::DADDInstruction::Flags::NEG_B));
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

}
}
