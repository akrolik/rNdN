#include "Backend/Codegen/Generators/Instructions/Arithmetic/MultiplyGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void MultiplyGenerator::Generate(const PTX::_MultiplyInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void MultiplyGenerator::Visit(const PTX::MultiplyInstruction<T> *instruction)
{
	// Types:
	//   - Int16, Int32, Int64
	//   - UInt16, UInt32, UInt64
	//   - Float16, Float16x2, Float32, Float64
	// Modifiers:
	//   - Half: Int16, Int32, Int64, UInt16, UInt32, UInt64
	//   - FlushSubnormal: Float16, Float16x2, Float32
	//   - Rounding: Float16, Float16x2, Float32, Float64
	//   - Saturate: Float16, Float16x2, Float32

	RegisterGenerator registerGenerator(this->m_builder);
	CompositeGenerator compositeGenerator(this->m_builder);
	compositeGenerator.SetImmediateValue(false);

	if constexpr(PTX::is_int_type<T>::value)
	{
		// Generate operands

		auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());
		auto [sourceA_Lo, sourceA_Hi] = registerGenerator.GeneratePair(instruction->GetSourceA());
		auto [sourceB_Lo, sourceB_Hi] = compositeGenerator.GeneratePair(instruction->GetSourceB());

		if constexpr(std::is_same<T, PTX::UInt32Type>::value || std::is_same<T, PTX::Int32Type>::value)
		{
			if (instruction->GetHalf() == PTX::MADInstruction<T>::Half::Lower)
			{
				// Compute D = (S1 * S2).lo
				//
				//   XMAD TMP0, S1, S2, RZ ;
				//   XMAD.MRG TMP1, S1, S2.H1, RZ ;
				//   XMAD.PSL.CBCC D, S1.H1, TMP1.H1, TMP0 ;

				auto temp0 = this->m_builder.AllocateTemporaryRegister();
				auto temp1 = this->m_builder.AllocateTemporaryRegister();

				this->AddInstruction(new SASS::XMADInstruction(temp0, sourceA_Lo, sourceB_Lo, SASS::RZ));
				this->AddInstruction(new SASS::XMADInstruction(
					temp1, sourceA_Lo, sourceB_Lo, SASS::RZ, SASS::XMADInstruction::Mode::MRG, SASS::XMADInstruction::Flags::H1_B
				));
				this->AddInstruction(new SASS::XMADInstruction(
					destination_Lo, sourceA_Lo, temp1, temp0, SASS::XMADInstruction::Mode::PSL,
					SASS::XMADInstruction::Flags::CBCC | SASS::XMADInstruction::Flags::H1_A | SASS::XMADInstruction::Flags::H1_B
				));
			}
			else
			{
				Error(instruction, "unsupported half modifier");
			}
		}
		else if constexpr(std::is_same<T, PTX::Int64Type>::value)
		{
			if (instruction->GetHalf() == PTX::MADInstruction<T>::Half::Lower)
			{
				// Compute D = (S1 * S2).lo

				auto temp0 = this->m_builder.AllocateTemporaryRegister();
				auto temp1 = this->m_builder.AllocateTemporaryRegister();
				auto temp2 = this->m_builder.AllocateTemporaryRegister();
				auto temp3 = this->m_builder.AllocateTemporaryRegister();
				auto temp4 = this->m_builder.AllocateTemporaryRegister();
				auto temp5 = this->m_builder.AllocateTemporaryRegister();
				auto temp6 = this->m_builder.AllocateTemporaryRegister();
				auto temp7 = this->m_builder.AllocateTemporaryRegister();
				auto temp8 = this->m_builder.AllocateTemporaryRegister();
				auto temp9 = this->m_builder.AllocateTemporaryRegister();

				// XMAD TMP0, S1, S2, RZ ;
				// XMAD TMP1, S1, S2.H1, RZ ;
				// XMAD TMP2, S1.H1, S2.H1, RZ ;

				this->AddInstruction(new SASS::XMADInstruction(temp0, sourceA_Lo, sourceB_Lo, SASS::RZ));
				this->AddInstruction(new SASS::XMADInstruction(
					temp1, sourceA_Lo, sourceB_Lo, SASS::RZ, SASS::XMADInstruction::Mode::None, SASS::XMADInstruction::Flags::H1_B
				));
				this->AddInstruction(new SASS::XMADInstruction(
					temp2, sourceA_Lo, sourceB_Lo, SASS::RZ, SASS::XMADInstruction::Mode::None,
					SASS::XMADInstruction::Flags::H1_A | SASS::XMADInstruction::Flags::H1_B
				));

				// XMAD TMP3, S1_HI, S2, RZ ;
				// XMAD.MRG TMP4, S1_HI, S2.H1, RZ ;
				// XMAD TMP5, S1, S2_HI, RZ ;

				this->AddInstruction(new SASS::XMADInstruction(temp3, sourceA_Hi, sourceB_Lo, SASS::RZ));
				this->AddInstruction(new SASS::XMADInstruction(
					temp4, sourceA_Hi, sourceB_Lo, SASS::RZ, SASS::XMADInstruction::Mode::MRG, SASS::XMADInstruction::Flags::H1_B
				));
				this->AddInstruction(new SASS::XMADInstruction(temp5, sourceA_Lo, sourceB_Hi, SASS::RZ));

				// XMAD.MRG TMP6, S1, S2_HI.H1, RZ ;
				// XMAD.CHI TMP7, S1.H1, S2, TMP0 ;
				// XMAD.MRG TMP8, S1, S2.H1, RZ ;

				this->AddInstruction(new SASS::XMADInstruction(
					temp6, sourceA_Lo, sourceB_Hi, SASS::RZ, SASS::XMADInstruction::Mode::MRG, SASS::XMADInstruction::Flags::H1_B
				));
				this->AddInstruction(new SASS::XMADInstruction(
					temp7, sourceA_Lo, sourceB_Lo, temp0, SASS::XMADInstruction::Mode::CHI, SASS::XMADInstruction::Flags::H1_A
				));
				this->AddInstruction(new SASS::XMADInstruction(
					temp8, sourceA_Lo, sourceB_Lo, SASS::RZ, SASS::XMADInstruction::Mode::MRG, SASS::XMADInstruction::Flags::H1_B
				));

				// XMAD.PSL.CBCC TMP9, S1_HI.H1, TMP4.H1, TMP3 ;
				// XMAD.PSL.CBCC TMP6, S1.H1, TMP6.H1, TMP5 ;
				// IADD3.RS TMP7, TMP7, TMP1, TMP2 ;

				this->AddInstruction(new SASS::XMADInstruction(
					temp9, sourceA_Hi, temp4, temp3, SASS::XMADInstruction::Mode::PSL,
					SASS::XMADInstruction::Flags::CBCC | SASS::XMADInstruction::Flags::H1_A | SASS::XMADInstruction::Flags::H1_B
				));
				this->AddInstruction(new SASS::XMADInstruction(
					temp6, sourceA_Lo, temp6, temp5, SASS::XMADInstruction::Mode::PSL,
					SASS::XMADInstruction::Flags::CBCC | SASS::XMADInstruction::Flags::H1_A | SASS::XMADInstruction::Flags::H1_B
				));
				this->AddInstruction(new SASS::IADD3Instruction(temp7, temp7, temp1, temp2, SASS::IADD3Instruction::Flags::RS));

				// XMAD.PSL.CBCC D, S1, TMP8.H1, TMP0 ;
				// IADD3 D_HI, TMP6, TMP7, TMP9 ;

				this->AddInstruction(new SASS::XMADInstruction(
					destination_Lo, sourceA_Lo, temp8, temp0, SASS::XMADInstruction::Mode::PSL,
					SASS::XMADInstruction::Flags::CBCC | SASS::XMADInstruction::Flags::H1_A | SASS::XMADInstruction::Flags::H1_B
				));
				this->AddInstruction(new SASS::IADD3Instruction(destination_Hi, temp6, temp7, temp9));
			}
			else
			{
				Error(instruction, "unsupported half modifier");
			}
		}
	}
	else if constexpr(std::is_same<T, PTX::Float64Type>::value)
	{
		// Generate operands

		auto destination = registerGenerator.Generate(instruction->GetDestination());
		auto sourceA = registerGenerator.Generate(instruction->GetSourceA());
		auto sourceB = compositeGenerator.Generate(instruction->GetSourceB());

		// Generate instruction

		auto round = SASS::DMULInstruction::Round::RN;
		switch (instruction->GetRoundingMode())
		{
			// case T::RoundingMode::None:
			// case T::RoundingMode::Nearest:
			// {
			// 	round = SASS::DMULInstruction::Round::RN;
			// 	break;
			// }
			case T::RoundingMode::Zero:
			{
				round = SASS::DMULInstruction::Round::RZ;
				break;
			}
			case T::RoundingMode::NegativeInfinity:
			{
				round = SASS::DMULInstruction::Round::RM;
				break;
			}
			case T::RoundingMode::PositiveInfinity:
			{
				round = SASS::DMULInstruction::Round::RP;
				break;
			}
		}

		this->AddInstruction(new SASS::DMULInstruction(destination, sourceA, sourceB, round));
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

}
}
