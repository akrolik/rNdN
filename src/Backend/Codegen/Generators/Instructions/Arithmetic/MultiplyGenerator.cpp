#include "Backend/Codegen/Generators/Instructions/Arithmetic/MultiplyGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void MultiplyGenerator::Generate(const PTX::_MultiplyInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void MultiplyGenerator::Visit(const PTX::MultiplyInstruction<T> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types:
	//   - Int16, Int32, Int64
	//   - UInt16, UInt32, UInt64
	//   - Float16, Float16x2, Float32, Float64
	// Modifiers:
	//   - Half: Int16, Int32, Int64, UInt16, UInt32, UInt64
	//   - FlushSubnormal: Float16, Float16x2, Float32
	//   - Rounding: Float16, Float16x2, Float32, Float64
	//   - Saturate: Float16, Float16x2, Float32

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<class T>
void MultiplyGenerator::GenerateMaxwell(const PTX::MultiplyInstruction<T> *instruction)
{
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

				this->AddInstruction(new SASS::Maxwell::XMADInstruction(temp0, sourceA_Lo, sourceB_Lo, SASS::RZ));
				this->AddInstruction(new SASS::Maxwell::XMADInstruction(
					temp1, sourceA_Lo, sourceB_Lo, SASS::RZ,
					SASS::Maxwell::XMADInstruction::Mode::MRG, SASS::Maxwell::XMADInstruction::Flags::H1_B
				));
				this->AddInstruction(new SASS::Maxwell::XMADInstruction(
					destination_Lo, sourceA_Lo, temp1, temp0, SASS::Maxwell::XMADInstruction::Mode::PSL,
					SASS::Maxwell::XMADInstruction::Flags::CBCC | SASS::Maxwell::XMADInstruction::Flags::H1_A |
					SASS::Maxwell::XMADInstruction::Flags::H1_B
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

				this->AddInstruction(new SASS::Maxwell::XMADInstruction(temp0, sourceA_Lo, sourceB_Lo, SASS::RZ));
				this->AddInstruction(new SASS::Maxwell::XMADInstruction(
					temp1, sourceA_Lo, sourceB_Lo, SASS::RZ,
					SASS::Maxwell::XMADInstruction::Mode::None, SASS::Maxwell::XMADInstruction::Flags::H1_B
				));
				this->AddInstruction(new SASS::Maxwell::XMADInstruction(
					temp2, sourceA_Lo, sourceB_Lo, SASS::RZ, SASS::Maxwell::XMADInstruction::Mode::None,
					SASS::Maxwell::XMADInstruction::Flags::H1_A | SASS::Maxwell::XMADInstruction::Flags::H1_B
				));

				// XMAD TMP3, S1_HI, S2, RZ ;
				// XMAD.MRG TMP4, S1_HI, S2.H1, RZ ;
				// XMAD TMP5, S1, S2_HI, RZ ;

				this->AddInstruction(new SASS::Maxwell::XMADInstruction(temp3, sourceA_Hi, sourceB_Lo, SASS::RZ));
				this->AddInstruction(new SASS::Maxwell::XMADInstruction(
					temp4, sourceA_Hi, sourceB_Lo, SASS::RZ,
					SASS::Maxwell::XMADInstruction::Mode::MRG, SASS::Maxwell::XMADInstruction::Flags::H1_B
				));
				this->AddInstruction(new SASS::Maxwell::XMADInstruction(temp5, sourceA_Lo, sourceB_Hi, SASS::RZ));

				// XMAD.MRG TMP6, S1, S2_HI.H1, RZ ;
				// XMAD.CHI TMP7, S1.H1, S2, TMP0 ;
				// XMAD.MRG TMP8, S1, S2.H1, RZ ;

				this->AddInstruction(new SASS::Maxwell::XMADInstruction(
					temp6, sourceA_Lo, sourceB_Hi, SASS::RZ,
					SASS::Maxwell::XMADInstruction::Mode::MRG, SASS::Maxwell::XMADInstruction::Flags::H1_B
				));
				this->AddInstruction(new SASS::Maxwell::XMADInstruction(
					temp7, sourceA_Lo, sourceB_Lo, temp0,
					SASS::Maxwell::XMADInstruction::Mode::CHI, SASS::Maxwell::XMADInstruction::Flags::H1_A
				));
				this->AddInstruction(new SASS::Maxwell::XMADInstruction(
					temp8, sourceA_Lo, sourceB_Lo, SASS::RZ,
					SASS::Maxwell::XMADInstruction::Mode::MRG, SASS::Maxwell::XMADInstruction::Flags::H1_B
				));

				// XMAD.PSL.CBCC TMP9, S1_HI.H1, TMP4.H1, TMP3 ;
				// XMAD.PSL.CBCC TMP6, S1.H1, TMP6.H1, TMP5 ;
				// IADD3.RS TMP7, TMP7, TMP1, TMP2 ;

				this->AddInstruction(new SASS::Maxwell::XMADInstruction(
					temp9, sourceA_Hi, temp4, temp3, SASS::Maxwell::XMADInstruction::Mode::PSL,
					SASS::Maxwell::XMADInstruction::Flags::CBCC | SASS::Maxwell::XMADInstruction::Flags::H1_A |
					SASS::Maxwell::XMADInstruction::Flags::H1_B
				));
				this->AddInstruction(new SASS::Maxwell::XMADInstruction(
					temp6, sourceA_Lo, temp6, temp5, SASS::Maxwell::XMADInstruction::Mode::PSL,
					SASS::Maxwell::XMADInstruction::Flags::CBCC | SASS::Maxwell::XMADInstruction::Flags::H1_A |
					SASS::Maxwell::XMADInstruction::Flags::H1_B
				));
				this->AddInstruction(new SASS::Maxwell::IADD3Instruction(
					temp7, temp7, temp1, temp2, SASS::Maxwell::IADD3Instruction::Flags::RS
				));

				// XMAD.PSL.CBCC D, S1, TMP8.H1, TMP0 ;
				// IADD3 D_HI, TMP6, TMP7, TMP9 ;

				this->AddInstruction(new SASS::Maxwell::XMADInstruction(
					destination_Lo, sourceA_Lo, temp8, temp0, SASS::Maxwell::XMADInstruction::Mode::PSL,
					SASS::Maxwell::XMADInstruction::Flags::CBCC | SASS::Maxwell::XMADInstruction::Flags::H1_A |
					SASS::Maxwell::XMADInstruction::Flags::H1_B
				));
				this->AddInstruction(new SASS::Maxwell::IADD3Instruction(destination_Hi, temp6, temp7, temp9));
			}
			else
			{
				Error(instruction, "unsupported half modifier");
			}
		}
		else
		{
			Error(instruction, "unsupported type");
		}
	}
	else if constexpr(std::is_same<T, PTX::Float64Type>::value)
	{
		// Generate operands

		auto destination = registerGenerator.Generate(instruction->GetDestination());
		auto sourceA = registerGenerator.Generate(instruction->GetSourceA());
		auto sourceB = compositeGenerator.Generate(instruction->GetSourceB());

		// Generate instruction

		auto round = SASS::Maxwell::DMULInstruction::Round::RN;
		switch (instruction->GetRoundingMode())
		{
			// case T::RoundingMode::None:
			// case T::RoundingMode::Nearest:
			// {
			// 	round = SASS::Maxwell::DMULInstruction::Round::RN;
			// 	break;
			// }
			case T::RoundingMode::Zero:
			{
				round = SASS::Maxwell::DMULInstruction::Round::RZ;
				break;
			}
			case T::RoundingMode::NegativeInfinity:
			{
				round = SASS::Maxwell::DMULInstruction::Round::RM;
				break;
			}
			case T::RoundingMode::PositiveInfinity:
			{
				round = SASS::Maxwell::DMULInstruction::Round::RP;
				break;
			}
		}

		this->AddInstruction(new SASS::Maxwell::DMULInstruction(destination, sourceA, sourceB, round));
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

template<class T>
void MultiplyGenerator::GenerateVolta(const PTX::MultiplyInstruction<T> *instruction)
{
	RegisterGenerator registerGenerator(this->m_builder);
	CompositeGenerator compositeGenerator(this->m_builder);
	compositeGenerator.SetImmediateSize(32);

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
				this->AddInstruction(new SASS::Volta::IMADInstruction(destination_Lo, sourceA_Lo, sourceB_Lo, SASS::RZ));
			}
			else if (instruction->GetHalf() == PTX::MADInstruction<T>::Half::Upper)
			{
				auto mode = SASS::Volta::IMADInstruction::Mode::HI;
				auto flags = SASS::Volta::IMADInstruction::Flags::None;
				if constexpr(std::is_same<T, PTX::UInt32Type>::value)
				{
					flags |= SASS::Volta::IMADInstruction::Flags::U32;
				}

				this->AddInstruction(new SASS::Volta::IMADInstruction(
					destination_Lo, sourceA_Lo, sourceB_Lo, SASS::RZ, mode, flags
				));
			}
		}
		else if constexpr(std::is_same<T, PTX::Int64Type>::value || std::is_same<T, PTX::UInt64Type>::value)
		{
			if (instruction->GetHalf() == PTX::MADInstruction<T>::Half::Lower)
			{
				// Compute D = (S1 * S2).lo
				//
				//   IMAD TMP, S1_HI, S2_LO, RZ
				//   IMAD TMP, S1_LO, S2_HI, TMP
				//   IMAD.WIDE.U32 D_LO, S1_LO, S2_LO, RZ
				//   IADD3 D_HI, D_HI, TMP, RZ

				// Wide IMAD uses both registers
				auto destination = registerGenerator.Generate(instruction->GetDestination());
				auto temp = this->m_builder.AllocateTemporaryRegister();

				this->AddInstruction(new SASS::Volta::IMADInstruction(temp, sourceA_Hi, sourceB_Lo, SASS::RZ));
				this->AddInstruction(new SASS::Volta::IMADInstruction(temp, sourceA_Lo, sourceB_Hi, temp));
				this->AddInstruction(new SASS::Volta::IMADInstruction(
					destination, sourceA_Lo, sourceB_Lo, SASS::RZ,
					SASS::Volta::IMADInstruction::Mode::WIDE,
					SASS::Volta::IMADInstruction::Flags::U32
				));
				this->AddInstruction(new SASS::Volta::IADD3Instruction(destination_Hi, destination_Hi, temp, SASS::RZ));
			}
			else
			{
				Error(instruction, "unsupported half modifier");
			}
		}
		else
		{
			Error(instruction, "unsupported type");
		}
	}
	else if constexpr(std::is_same<T, PTX::Float64Type>::value)
	{
		// Generate operands

		auto destination = registerGenerator.Generate(instruction->GetDestination());
		auto sourceA = registerGenerator.Generate(instruction->GetSourceA());
		auto sourceB = compositeGenerator.Generate(instruction->GetSourceB());

		// Generate instruction

		auto round = SASS::Volta::DMULInstruction::Round::RN;
		switch (instruction->GetRoundingMode())
		{
			// case T::RoundingMode::None:
			// case T::RoundingMode::Nearest:
			// {
			// 	round = SASS::Volta::DMULInstruction::Round::RN;
			// 	break;
			// }
			case T::RoundingMode::Zero:
			{
				round = SASS::Volta::DMULInstruction::Round::RZ;
				break;
			}
			case T::RoundingMode::NegativeInfinity:
			{
				round = SASS::Volta::DMULInstruction::Round::RM;
				break;
			}
			case T::RoundingMode::PositiveInfinity:
			{
				round = SASS::Volta::DMULInstruction::Round::RP;
				break;
			}
		}

		this->AddInstruction(new SASS::Volta::DMULInstruction(destination, sourceA, sourceB, round));
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

}
}
