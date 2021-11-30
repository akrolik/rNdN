#include "Backend/Codegen/Generators/Instructions/Arithmetic/RemainderGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

#include "Utils/Math.h"

namespace Backend {
namespace Codegen {

void RemainderGenerator::Generate(const PTX::_RemainderInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void RemainderGenerator::Visit(const PTX::RemainderInstruction<T> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types:
	//   - Int16, Int32, Int64
	//   - UInt16, UInt32, UInt64
	// Modifiers: --

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<class T>
void RemainderGenerator::GenerateMaxwell(const PTX::RemainderInstruction<T> *instruction)
{
	// Generate operands

	RegisterGenerator registerGenerator(this->m_builder);
	CompositeGenerator compositeGenerator(this->m_builder);

	auto destination = registerGenerator.Generate(instruction->GetDestination());
	auto sourceA = registerGenerator.Generate(instruction->GetSourceA());
	auto sourceB = compositeGenerator.Generate(instruction->GetSourceB());

	// Generate instruction

	if constexpr(PTX::is_int_type<T>::value)
	{
		// Use 32-bit remainder for 16-bits too 

		if constexpr(T::TypeBits == PTX::Bits::Bits16 || T::TypeBits == PTX::Bits::Bits32)
		{
			// Optimize power of 2 remainder using bitwise &(divisor-1)

			if (auto immediateSourceB = dynamic_cast<SASS::I32Immediate *>(sourceB))
			{
				auto value = immediateSourceB->GetValue();
				if (value == Utils::Math::Power2(value))
				{
					immediateSourceB->SetValue(value - 1);
					this->AddInstruction(new SASS::Maxwell::LOP32IInstruction(
						destination, sourceA, immediateSourceB, SASS::Maxwell::LOP32IInstruction::BooleanOperator::AND
					));
					return;
				}
			}

			auto sourceB = registerGenerator.Generate(instruction->GetSourceB());

			// Compute D = S1 % S2

			auto temp0 = this->m_builder.AllocateTemporaryRegister();
			auto temp1 = this->m_builder.AllocateTemporaryRegister();
			auto temp2 = this->m_builder.AllocateTemporaryRegister();
			auto temp3 = this->m_builder.AllocateTemporaryRegister();
			auto temp4 = this->m_builder.AllocateTemporaryRegister();
			auto temp5 = this->m_builder.AllocateTemporaryRegister();
			auto temp6 = this->m_builder.AllocateTemporaryRegister();
			auto temp7 = this->m_builder.AllocateTemporaryRegister();

			auto pred = this->m_builder.AllocateTemporaryPredicate();

			// I2F.F32.U32.RP TMP0, S2 ;
			// MUFU.RCP TMP0, TMP0 ;
			// IADD32I TMP1, TMP0, 0xffffffe ;
			// F2I.FTZ.U32.F32.TRUNC TMP2, TMP1 ;

			this->AddInstruction(new SASS::Maxwell::I2FInstruction(
				temp0, sourceB, SASS::Maxwell::I2FInstruction::DestinationType::F32,
				SASS::Maxwell::I2FInstruction::SourceType::U32, SASS::Maxwell::I2FInstruction::Round::RP
			));
			this->AddInstruction(new SASS::Maxwell::MUFUInstruction(temp0, temp0, SASS::Maxwell::MUFUInstruction::Function::RCP));
			this->AddInstruction(new SASS::Maxwell::IADD32IInstruction(temp1, temp0, new SASS::I32Immediate(0xffffffe)));
			this->AddInstruction(new SASS::Maxwell::F2IInstruction(
				temp2, temp1, SASS::Maxwell::F2IInstruction::DestinationType::U32, SASS::Maxwell::F2IInstruction::SourceType::F32,
				SASS::Maxwell::F2IInstruction::Round::TRUNC, SASS::Maxwell::F2IInstruction::Flags::FTZ
			));

			// XMAD TMP3, TMP2.reuse, S2.reuse, RZ ;
			// XMAD.MRG TMP4, TMP2.reuse, S2.H1, RZ ;
			// XMAD.PSL.CBCC TMP3, TMP2.H1.reuse, TMP4.H1, TMP3 ;

			this->AddInstruction(new SASS::Maxwell::XMADInstruction(temp3, temp2, sourceB, SASS::RZ));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp4, temp2, sourceB, SASS::RZ, SASS::Maxwell::XMADInstruction::Mode::MRG, SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp3, temp2, temp4, temp3, SASS::Maxwell::XMADInstruction::Mode::PSL,
				SASS::Maxwell::XMADInstruction::Flags::H1_A | SASS::Maxwell::XMADInstruction::Flags::H1_B |
				SASS::Maxwell::XMADInstruction::Flags::CBCC
			));

			// IADD TMP5, -TMP3, RZ ;

			this->AddInstruction(new SASS::Maxwell::IADDInstruction(temp5, temp3, SASS::RZ, SASS::Maxwell::IADDInstruction::Flags::NEG_A));

			// XMAD TMP3, TMP2.reuse, TMP5.reuse, RZ ;
			// XMAD TMP6, TMP2, TMP5.H1, RZ ;
			// XMAD TMP4, TMP2.H1.reuse, TMP5.H1.reuse, TMP2 ;
			// XMAD.CHI TMP3, TMP2.H1, TMP5, TMP3 ;
			// IADD3.RS TMP6, TMP3, TMP6, TMP4 ;

			this->AddInstruction(new SASS::Maxwell::XMADInstruction(temp3, temp2, temp5, SASS::RZ));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp6, temp2, temp5, SASS::RZ, SASS::Maxwell::XMADInstruction::Mode::None, SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp4, temp2, temp5, temp2, SASS::Maxwell::XMADInstruction::Mode::None,
				SASS::Maxwell::XMADInstruction::Flags::H1_A | SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp3, temp2, temp5, temp3, SASS::Maxwell::XMADInstruction::Mode::CHI, SASS::Maxwell::XMADInstruction::Flags::H1_A
			));
			this->AddInstruction(new SASS::Maxwell::IADD3Instruction(
				temp6, temp3, temp6, temp4, SASS::Maxwell::IADD3Instruction::Flags::RS
			));

			// XMAD TMP0, TMP6.reuse, S1.reuse, RZ ;
			// XMAD TMP1, TMP6.reuse, S1.H1.reuse, RZ ;
			// XMAD TMP2, TMP6.H1, S1.H1, RZ ;
			// XMAD.CHI TMP0, TMP6.H1, S1, TMP0 ;
			// IADD3.RS TMP6, TMP0, TMP1, TMP2 ;

			this->AddInstruction(new SASS::Maxwell::XMADInstruction(temp0, temp6, sourceA, SASS::RZ));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp1, temp6, sourceA, SASS::RZ,
				SASS::Maxwell::XMADInstruction::Mode::None, SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp2, temp6, sourceA, SASS::RZ, SASS::Maxwell::XMADInstruction::Mode::None,
				SASS::Maxwell::XMADInstruction::Flags::H1_A | SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp0, temp6, sourceA, temp0,
				SASS::Maxwell::XMADInstruction::Mode::CHI, SASS::Maxwell::XMADInstruction::Flags::H1_A
			));
			this->AddInstruction(new SASS::Maxwell::IADD3Instruction(
				temp6, temp0, temp1, temp2, SASS::Maxwell::IADD3Instruction::Flags::RS
			));

			// IADD TMP6, -TMP6, RZ ;

			this->AddInstruction(new SASS::Maxwell::IADDInstruction(
				temp6, temp6, SASS::RZ, SASS::Maxwell::IADDInstruction::Flags::NEG_A
			));

			// XMAD TMP0, TMP6.reuse, S2.reuse, S1 ;
			// XMAD.MRG TMP1, TMP6.reuse, S2.H1, RZ ;
			// XMAD.PSL.CBCC D, TMP6.H1, TMP1.H1, TMP0 ;

			this->AddInstruction(new SASS::Maxwell::XMADInstruction(temp0, temp6, sourceB, sourceA));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp1, temp6, sourceB, SASS::RZ,
				SASS::Maxwell::XMADInstruction::Mode::MRG, SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp7, temp6, temp1, temp0, SASS::Maxwell::XMADInstruction::Mode::PSL,
				SASS::Maxwell::XMADInstruction::Flags::H1_A | SASS::Maxwell::XMADInstruction::Flags::H1_B |
				SASS::Maxwell::XMADInstruction::Flags::CBCC
			));

			//     ISETP.GE.U32.AND P0, PT, D, S2, PT ;
			// @P0 IADD D, -S2, D ;

			this->AddInstruction(new SASS::Maxwell::ISETPInstruction(
				pred, SASS::PT, temp7, sourceB, SASS::PT,
				SASS::Maxwell::ISETPInstruction::ComparisonOperator::GE,
				SASS::Maxwell::ISETPInstruction::BooleanOperator::AND,
				SASS::Maxwell::ISETPInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Maxwell::IADDInstruction(temp7, sourceB, temp7, SASS::Maxwell::IADDInstruction::Flags::NEG_A), pred);

			//     ISETP.GE.U32.AND P0, PT, D, S2, PT ;
			// @P0 IADD D, -S2, D ;

			this->AddInstruction(new SASS::Maxwell::ISETPInstruction(
				pred, SASS::PT, temp7, sourceB, SASS::PT,
				SASS::Maxwell::ISETPInstruction::ComparisonOperator::GE,
				SASS::Maxwell::ISETPInstruction::BooleanOperator::AND,
				SASS::Maxwell::ISETPInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Maxwell::IADDInstruction(temp7, sourceB, temp7, SASS::Maxwell::IADDInstruction::Flags::NEG_A), pred);

			//     ISETP.EQ.U32.AND P0, PT, S2, RZ, PT ;
			// @P0 LOP.PASS_B D, RZ, ~S2 ;

			this->AddInstruction(new SASS::Maxwell::ISETPInstruction(
				pred, SASS::PT, sourceB, SASS::RZ, SASS::PT,
				SASS::Maxwell::ISETPInstruction::ComparisonOperator::EQ,
				SASS::Maxwell::ISETPInstruction::BooleanOperator::AND,
				SASS::Maxwell::ISETPInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Maxwell::LOPInstruction(
				temp7, SASS::RZ, sourceB, SASS::Maxwell::LOPInstruction::BooleanOperator::PASS_B,
				SASS::Maxwell::LOPInstruction::Flags::INV
			), pred);

			// Result

			this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination, temp7));
		}
		else
		{
			Error(instruction, "unsupported non-constant type");
		}
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

template<class T>
void RemainderGenerator::GenerateVolta(const PTX::RemainderInstruction<T> *instruction)
{
	// Generate operands

	RegisterGenerator registerGenerator(this->m_builder);
	CompositeGenerator compositeGenerator(this->m_builder);
	compositeGenerator.SetImmediateSize(32);

	auto destination = registerGenerator.Generate(instruction->GetDestination());
	auto sourceA = registerGenerator.Generate(instruction->GetSourceA());
	auto sourceB = compositeGenerator.Generate(instruction->GetSourceB());

	// Generate instruction

	if constexpr(PTX::is_int_type<T>::value)
	{
		// Use 32-bit remainder for 16-bits too 

		if constexpr(T::TypeBits == PTX::Bits::Bits16 || T::TypeBits == PTX::Bits::Bits32)
		{
			// Optimize power of 2 remainder using bitwise &(divisor-1)

			if (auto immediateSourceB = dynamic_cast<SASS::I32Immediate *>(sourceB))
			{
				auto value = immediateSourceB->GetValue();
				if (value == Utils::Math::Power2(value))
				{
					immediateSourceB->SetValue(value - 1);

					auto logicOperation = SASS::Volta::BinaryUtils::LogicOperation(
						[](std::uint8_t A, std::uint8_t B, std::uint8_t C)
						{
							return ((A & B) | C);
						}
					);

					this->AddInstruction(new SASS::Volta::LOP3Instruction(
						destination, sourceA, sourceB, SASS::RZ, new SASS::I8Immediate(logicOperation), SASS::PT
					));
					return;
				}
			}

			auto sourceB = registerGenerator.Generate(instruction->GetSourceB());

			auto [temp1, temp2] = this->m_builder.AllocateTemporaryRegisterPair<PTX::Bits::Bits64>(); // Paired for IMAD.HI
			auto temp3 = this->m_builder.AllocateTemporaryRegister();

			auto pred = this->m_builder.AllocateTemporaryPredicate();

			// I2F.U32.RP TMP1, S2 ;
			// IMAD.MOV TMP3, RZ, RZ, -S2 ;
			// MUFU.RCP TMP1, TMP1 ;

			this->AddInstruction(new SASS::Volta::I2FInstruction(
				temp1, sourceB, SASS::Volta::I2FInstruction::DestinationType::F32, SASS::Volta::I2FInstruction::SourceType::U32,
				SASS::Volta::I2FInstruction::Round::RP
			));
			this->AddInstruction(new SASS::Volta::IMADInstruction(
				temp3, SASS::RZ, SASS::RZ, sourceB, SASS::Volta::IMADInstruction::Mode::Default, SASS::Volta::IMADInstruction::Flags::NEG_C
			));
			this->AddInstruction(new SASS::Volta::MUFUInstruction(temp1, temp1, SASS::Volta::MUFUInstruction::Function::RCP));

			// IADD3 TMP1, TMP1, 0xffffffe, RZ ;
			// F2I.FTZ.U32.TRUNC.NTZ TMP2, TMP1 ;

			this->AddInstruction(new SASS::Volta::IADD3Instruction(temp1, temp1, new SASS::I32Immediate(0xffffffe), SASS::RZ));
			this->AddInstruction(new SASS::Volta::F2IInstruction(
				temp2, temp1, SASS::Volta::F2IInstruction::DestinationType::U32, SASS::Volta::F2IInstruction::SourceType::F32,
				SASS::Volta::F2IInstruction::Round::TRUNC,
				SASS::Volta::F2IInstruction::Flags::FTZ | SASS::Volta::F2IInstruction::Flags::NTZ
			));

			// IMAD TMP3, TMP3, TMP2, RZ ;
			// IMAD.HI.U32 TMP2, TMP2, TMP3, TMP1 ;
			// IMAD.HI.U32 TMP2, TMP2, S2, RZ ;
			// IMAD.MOV TMP2, RZ, RZ, -TMP2 ;
			// IMAD D, S2, TMP2, S1 ;

			this->AddInstruction(new SASS::Volta::MOVInstruction(temp1, SASS::RZ));
			this->AddInstruction(new SASS::Volta::IMADInstruction(temp3, temp3, temp2, SASS::RZ));
			this->AddInstruction(new SASS::Volta::IMADInstruction(
				temp2, temp2, temp3, temp1, SASS::Volta::IMADInstruction::Mode::HI, SASS::Volta::IMADInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Volta::IMADInstruction(
				temp2, temp2, sourceA, SASS::RZ, SASS::Volta::IMADInstruction::Mode::HI, SASS::Volta::IMADInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Volta::IMADInstruction(
				temp2, SASS::RZ, SASS::RZ, temp2, SASS::Volta::IMADInstruction::Mode::Default, SASS::Volta::IMADInstruction::Flags::NEG_C
			));
			this->AddInstruction(new SASS::Volta::IMADInstruction(destination, sourceB, temp2, sourceA));

			// ISETP.GE.U32.AND PTMP, PT, D, S2, PT ;
			// @PTMP IADD3 D, -S2, D, RZ ;

			this->AddInstruction(new SASS::Volta::ISETPInstruction(
				pred, SASS::PT, destination, sourceB, SASS::PT,
				SASS::Volta::ISETPInstruction::ComparisonOperator::GE,
				SASS::Volta::ISETPInstruction::BooleanOperator::AND,
				SASS::Volta::ISETPInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Volta::IADD3Instruction(
				destination, sourceB, destination, SASS::RZ, SASS::Volta::IADD3Instruction::Flags::NEG_A
			), pred);

			// ISETP.GE.U32.AND PTMP, PT, D, S2, PT ;
			// @PTMP IADD3 D, -S2, D, RZ ;

			this->AddInstruction(new SASS::Volta::ISETPInstruction(
				pred, SASS::PT, destination, sourceB, SASS::PT,
				SASS::Volta::ISETPInstruction::ComparisonOperator::GE,
				SASS::Volta::ISETPInstruction::BooleanOperator::AND,
				SASS::Volta::ISETPInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Volta::IADD3Instruction(
				destination, sourceB, destination, SASS::RZ, SASS::Volta::IADD3Instruction::Flags::NEG_A
			), pred); 

			// ISETP.EQ.U32.AND PTMP, PT, S2, RZ, PT ;
			// @PTMP LOP3.LUT D, RZ, S2, RZ, 0x33, !PT ;

			this->AddInstruction(new SASS::Volta::ISETPInstruction(
				pred, SASS::PT, sourceB, SASS::RZ, SASS::PT,
				SASS::Volta::ISETPInstruction::ComparisonOperator::EQ,
				SASS::Volta::ISETPInstruction::BooleanOperator::AND,
				SASS::Volta::ISETPInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Volta::LOP3Instruction(
				destination, SASS::RZ, sourceB, SASS::RZ, new SASS::I8Immediate(0x33), SASS::PT, SASS::Volta::LOP3Instruction::Flags::NOT_E
			), pred);
		}
		else
		{
			Error(instruction, "unsupported non-constant type");
		}
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

}
}
