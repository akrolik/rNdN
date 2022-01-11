#include "Backend/Codegen/Generators/Instructions/Arithmetic/DivideGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void DivideGenerator::Generate(const PTX::_DivideInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void DivideGenerator::Visit(const PTX::DivideInstruction<T> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types:
	//   - Int16, Int32, Int64
	//   - UInt16, UInt32, UInt64
	//   - Float32, Float64
	// Modifiers:
	//   - FlushSubnormal: Float32
	//   - Rounding: Float32, Float64

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<class T>
void DivideGenerator::GenerateMaxwell(const PTX::DivideInstruction<T> *instruction)
{
	if constexpr(PTX::is_int_type<T>::value)
	{
		// Generate operands

		RegisterGenerator registerGenerator(this->m_builder);
		CompositeGenerator compositeGenerator(this->m_builder);

		auto destination = registerGenerator.Generate(instruction->GetDestination());
		auto sourceA = registerGenerator.Generate(instruction->GetSourceA());
		auto sourceB = compositeGenerator.Generate(instruction->GetSourceB());

		if constexpr(T::TypeBits == PTX::Bits::Bits32)
		{
			if (auto immediateSourceB = dynamic_cast<SASS::I32Immediate *>(sourceB))
			{
				// Special case for power2 division

				auto value = immediateSourceB->GetValue();
				if (value == Utils::Math::Power2(value))
				{
					auto logValue = Utils::Math::Log2(value);
					immediateSourceB->SetValue(logValue);
					
					auto flags = SASS::Maxwell::SHRInstruction::Flags::None;
					if constexpr(PTX::is_unsigned_int_type<T>::value)
					{
						flags |= SASS::Maxwell::SHRInstruction::Flags::U32;
					}

					this->AddInstruction(new SASS::Maxwell::SHRInstruction(destination, sourceA, immediateSourceB, flags));
					return;
				}
			}
		}

		// Generate instruction

		if constexpr(std::is_same<T, PTX::UInt32Type>::value)
		{
			auto sourceB = registerGenerator.Generate(instruction->GetSourceB());

			// Compute D = S1 / S2

			auto temp0 = this->m_builder.AllocateTemporaryRegister();
			auto temp1 = this->m_builder.AllocateTemporaryRegister();
			auto temp2 = this->m_builder.AllocateTemporaryRegister();
			auto temp3 = this->m_builder.AllocateTemporaryRegister();
			auto temp4 = this->m_builder.AllocateTemporaryRegister();
			auto temp5 = this->m_builder.AllocateTemporaryRegister();

			auto pred = this->m_builder.AllocateTemporaryPredicate();

			// I2F.F32.U32.RP TMP0, S2 ;
			// MUFU.RCP TMP0, TMP0 ;
			// IADD32I TMP0, TMP0, 0xffffffe ;
			// F2I.FTZ.U32.F32.TRUNC TMP0, TMP0 ;

			this->AddInstruction(new SASS::Maxwell::I2FInstruction(
				temp0, sourceB, SASS::Maxwell::I2FInstruction::DestinationType::F32, SASS::Maxwell::I2FInstruction::SourceType::U32,
				SASS::Maxwell::I2FInstruction::Round::RP
			));
			this->AddInstruction(new SASS::Maxwell::MUFUInstruction(temp0, temp0, SASS::Maxwell::MUFUInstruction::Function::RCP));
			this->AddInstruction(new SASS::Maxwell::IADD32IInstruction(temp0, temp0, new SASS::I32Immediate(0xffffffe)));
			this->AddInstruction(new SASS::Maxwell::F2IInstruction(
				temp0, temp0, SASS::Maxwell::F2IInstruction::DestinationType::U32, SASS::Maxwell::F2IInstruction::SourceType::F32,
				SASS::Maxwell::F2IInstruction::Round::TRUNC, SASS::Maxwell::F2IInstruction::Flags::FTZ
			));

			// XMAD TMP1, TMP0, S2, RZ ;
			// XMAD.MRG TMP2, TMP0, S2.H1, RZ ;
			// XMAD.PSL.CBCC TMP1, TMP0.H1, TMP2.H1, TMP1 ;

			this->AddInstruction(new SASS::Maxwell::XMADInstruction(temp1, temp0, sourceB, SASS::RZ));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp2, temp0, sourceB, SASS::RZ, SASS::Maxwell::XMADInstruction::Mode::MRG, SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp1, temp0, temp2, temp1, SASS::Maxwell::XMADInstruction::Mode::PSL,
				SASS::Maxwell::XMADInstruction::Flags::H1_A | SASS::Maxwell::XMADInstruction::Flags::H1_B | SASS::Maxwell::XMADInstruction::Flags::CBCC
			));

			// IADD TMP1, -TMP1, RZ ;

			this->AddInstruction(new SASS::Maxwell::IADDInstruction(temp1, temp1, SASS::RZ, SASS::Maxwell::IADDInstruction::Flags::NEG_A));

			// XMAD TMP2, TMP0, TMP1, RZ ;
			// XMAD TMP3, TMP0, TMP1.H1, RZ ;
			// XMAD TMP4, TMP0.H1, TMP1.H1, TMP0 ;
			// XMAD.CHI TMP2, TMP0.H1, TMP1, TMP2 ;
			// IADD3.RS TMP3, TMP2, TMP3, TMP4 ;

			this->AddInstruction(new SASS::Maxwell::XMADInstruction(temp2, temp0, temp1, SASS::RZ));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp3, temp0, temp1, SASS::RZ, SASS::Maxwell::XMADInstruction::Mode::None, SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp4, temp0, temp1, temp0, SASS::Maxwell::XMADInstruction::Mode::None,
				SASS::Maxwell::XMADInstruction::Flags::H1_A | SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp2, temp0, temp1, temp2, SASS::Maxwell::XMADInstruction::Mode::CHI, SASS::Maxwell::XMADInstruction::Flags::H1_A
			));
			this->AddInstruction(new SASS::Maxwell::IADD3Instruction(
				temp3, temp2, temp3, temp4, SASS::Maxwell::IADD3Instruction::Flags::RS
			));

			// XMAD TMP0, TMP3, S1, RZ ;
			// XMAD TMP1, TMP3, S1.H1, RZ ;
			// XMAD TMP2, TMP3.H1, S1.H1, RZ ;
			// XMAD.CHI TMP0, TMP3.H1, S1, TMP0 ;
			// IADD3.RS D, TMP0, TMP1, TMP2 ;

			this->AddInstruction(new SASS::Maxwell::XMADInstruction(temp0, temp3, sourceA, SASS::RZ));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp1, temp3, sourceA, SASS::RZ, SASS::Maxwell::XMADInstruction::Mode::None, SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp2, temp3, sourceA, SASS::RZ, SASS::Maxwell::XMADInstruction::Mode::None,
				SASS::Maxwell::XMADInstruction::Flags::H1_A | SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp0, temp3, sourceA, temp0, SASS::Maxwell::XMADInstruction::Mode::CHI, SASS::Maxwell::XMADInstruction::Flags::H1_A
			));
			this->AddInstruction(new SASS::Maxwell::IADD3Instruction(
				temp5, temp0, temp1, temp2, SASS::Maxwell::IADD3Instruction::Flags::RS
			));

			// IADD TMP0, -D, RZ ;

			this->AddInstruction(new SASS::Maxwell::IADDInstruction(temp0, temp5, SASS::RZ, SASS::Maxwell::IADDInstruction::Flags::NEG_A));

			// XMAD TMP1, TMP0, S2, S1 ;
			// XMAD.MRG TMP2, TMP0, S2.H1, RZ ;
			// XMAD.PSL.CBCC TMP0, TMP0.H1, TMP2.H1, TMP1 ;

			this->AddInstruction(new SASS::Maxwell::XMADInstruction(temp1, temp0, sourceB, sourceA));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp2, temp0, sourceB, SASS::RZ, SASS::Maxwell::XMADInstruction::Mode::MRG, SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp0, temp0, temp2, temp1, SASS::Maxwell::XMADInstruction::Mode::PSL,
				SASS::Maxwell::XMADInstruction::Flags::H1_A | SASS::Maxwell::XMADInstruction::Flags::H1_B | SASS::Maxwell::XMADInstruction::Flags::CBCC
			));

			//     ISETP.GE.U32.AND P0, PT, TMP0, S2, PT ;
			// @P0 IADD TMP0, -S2, TMP0 ;
			// @P0 IADD32I TMP5, TMP5, 0x1 ;

			this->AddInstruction(new SASS::Maxwell::ISETPInstruction(
				pred, SASS::PT, temp0, sourceB, SASS::PT,
				SASS::Maxwell::ISETPInstruction::ComparisonOperator::GE,
				SASS::Maxwell::ISETPInstruction::BooleanOperator::AND,
				SASS::Maxwell::ISETPInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Maxwell::IADDInstruction(temp0, sourceB, temp0, SASS::Maxwell::IADDInstruction::Flags::NEG_A), pred);
			this->AddInstruction(new SASS::Maxwell::IADD32IInstruction(temp5, temp5, new SASS::I32Immediate(0x1)), pred);

			//     ISETP.GE.U32.AND P0, PT, TMP0, S2, PT ;
			// @P0 IADD32I TMP5, TMP5, 0x1 ;

			this->AddInstruction(new SASS::Maxwell::ISETPInstruction(
				pred, SASS::PT, temp0, sourceB, SASS::PT,
				SASS::Maxwell::ISETPInstruction::ComparisonOperator::GE,
				SASS::Maxwell::ISETPInstruction::BooleanOperator::AND,
				SASS::Maxwell::ISETPInstruction::Flags::U32
			));

			this->AddInstruction(new SASS::Maxwell::IADD32IInstruction(temp5, temp5, new SASS::I32Immediate(0x1)), pred);

			//     ISETP.EQ.U32.AND P0, PT, S2, RZ, PT ;
			// @P0 LOP.PASS_B TMP5, RZ, ~S2 ;

			this->AddInstruction(new SASS::Maxwell::ISETPInstruction(
				pred, SASS::PT, sourceB, SASS::RZ, SASS::PT,
				SASS::Maxwell::ISETPInstruction::ComparisonOperator::EQ,
				SASS::Maxwell::ISETPInstruction::BooleanOperator::AND,
				SASS::Maxwell::ISETPInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Maxwell::LOPInstruction(
				temp5, SASS::RZ, sourceB, SASS::Maxwell::LOPInstruction::BooleanOperator::PASS_B, SASS::Maxwell::LOPInstruction::Flags::INV
			), pred);

			// Result

			this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination, temp5));
		}
		else if constexpr(std::is_same<T, PTX::Int32Type>::value)
		{
			auto sourceB = registerGenerator.Generate(instruction->GetSourceB());

			// Compute D = S1 / S2

			auto temp0 = this->m_builder.AllocateTemporaryRegister();
			auto temp1 = this->m_builder.AllocateTemporaryRegister();
			auto temp2 = this->m_builder.AllocateTemporaryRegister();
			auto temp3 = this->m_builder.AllocateTemporaryRegister();
			auto temp4 = this->m_builder.AllocateTemporaryRegister();
			auto temp5 = this->m_builder.AllocateTemporaryRegister();
			auto temp6 = this->m_builder.AllocateTemporaryRegister();

			auto pred = this->m_builder.AllocateTemporaryPredicate();

			// I2F.F32.S32.RP TMP0, |S2| ;
			// I2I.S32.S32 TMP1, |S2| ;
			// I2I.S32.S32 TMP2, |S1| ;

			this->AddInstruction(new SASS::Maxwell::I2FInstruction(
				temp0, sourceB, SASS::Maxwell::I2FInstruction::DestinationType::F32, SASS::Maxwell::I2FInstruction::SourceType::S32,
				SASS::Maxwell::I2FInstruction::Round::RP, SASS::Maxwell::I2FInstruction::Flags::ABS
			));
			this->AddInstruction(new SASS::Maxwell::I2IInstruction(
				temp1, sourceB, SASS::Maxwell::I2IInstruction::DestinationType::S32, SASS::Maxwell::I2IInstruction::SourceType::S32,
				SASS::Maxwell::I2IInstruction::Flags::ABS
			));
			this->AddInstruction(new SASS::Maxwell::I2IInstruction(
				temp2, sourceA, SASS::Maxwell::I2IInstruction::DestinationType::S32, SASS::Maxwell::I2IInstruction::SourceType::S32,
				SASS::Maxwell::I2IInstruction::Flags::ABS
			));

			// MUFU.RCP TMP0, TMP0 ;
			// IADD32I TMP3, TMP0, 0xffffffe ;
			// F2I.FTZ.U32.F32.TRUNC TMP3, TMP3 ;

			this->AddInstruction(new SASS::Maxwell::MUFUInstruction(temp0, temp0, SASS::Maxwell::MUFUInstruction::Function::RCP));
			this->AddInstruction(new SASS::Maxwell::IADD32IInstruction(temp3, temp0, new SASS::I32Immediate(0xffffffe)));
			this->AddInstruction(new SASS::Maxwell::F2IInstruction(
				temp3, temp3, SASS::Maxwell::F2IInstruction::DestinationType::U32, SASS::Maxwell::F2IInstruction::SourceType::F32,
				SASS::Maxwell::F2IInstruction::Round::TRUNC, SASS::Maxwell::F2IInstruction::Flags::FTZ
			));

			// XMAD TMP5, TMP1, TMP3, RZ ;
			// XMAD.MRG TMP4, TMP1, TMP3.H1, RZ ;
			// XMAD.PSL.CBCC TMP5, TMP1.H1, TMP4.H1, TMP5 ;
			// IADD TMP6, -TMP5, RZ ;

			this->AddInstruction(new SASS::Maxwell::XMADInstruction(temp5, temp1, temp3, SASS::RZ));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp4, temp1, temp3, SASS::RZ, SASS::Maxwell::XMADInstruction::Mode::MRG, SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp5, temp1, temp4, temp5, SASS::Maxwell::XMADInstruction::Mode::PSL,
				SASS::Maxwell::XMADInstruction::Flags::CBCC | SASS::Maxwell::XMADInstruction::Flags::H1_A | SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::IADDInstruction(temp6, temp5, SASS::RZ, SASS::Maxwell::IADDInstruction::Flags::NEG_A));

			// XMAD TMP5, TMP3, TMP6, RZ ;
			// XMAD TMP4, TMP3, TMP6.H1, RZ ;
			// XMAD TMP0, TMP3.H1, TMP6.H1, TMP3 ;
			// XMAD.CHI TMP5, TMP3.H1, TMP6, TMP5 ;
			// IADD3.RS TMP0, TMP5, TMP4, TMP0 ;

			this->AddInstruction(new SASS::Maxwell::XMADInstruction(temp5, temp3, temp6, SASS::RZ));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp4, temp3, temp6, SASS::RZ, SASS::Maxwell::XMADInstruction::Mode::None, SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp0, temp3, temp6, temp3, SASS::Maxwell::XMADInstruction::Mode::None,
				SASS::Maxwell::XMADInstruction::Flags::H1_A | SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp5, temp3, temp6, temp5, SASS::Maxwell::XMADInstruction::Mode::CHI, SASS::Maxwell::XMADInstruction::Flags::H1_A
			));
			this->AddInstruction(new SASS::Maxwell::IADD3Instruction(temp0, temp5, temp4, temp0, SASS::Maxwell::IADD3Instruction::Flags::RS));

			// XMAD TMP3, TMP0, TMP2, RZ ;
			// XMAD TMP5, TMP0, TMP2.H1, RZ ;
			// XMAD.CHI TMP3, TMP0.H1, TMP2, TMP3 ;
			// XMAD TMP0, TMP0.H1, TMP2.H1, RZ ;
			// IADD3.RS TMP0, TMP3, TMP5, TMP0 ;

			this->AddInstruction(new SASS::Maxwell::XMADInstruction(temp3, temp0, temp2, SASS::RZ));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp5, temp0, temp2, SASS::RZ, SASS::Maxwell::XMADInstruction::Mode::None, SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp3, temp0, temp2, temp3, SASS::Maxwell::XMADInstruction::Mode::CHI, SASS::Maxwell::XMADInstruction::Flags::H1_A
			));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp0, temp0, temp2, SASS::RZ, SASS::Maxwell::XMADInstruction::Mode::None,
				SASS::Maxwell::XMADInstruction::Flags::H1_A | SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::IADD3Instruction(temp0, temp3, temp5, temp0, SASS::Maxwell::IADD3Instruction::Flags::RS));

			// I2I.S32.S32 TMP4, -|S2| ;

			this->AddInstruction(new SASS::Maxwell::I2IInstruction(
				temp4, sourceB, SASS::Maxwell::I2IInstruction::DestinationType::S32, SASS::Maxwell::I2IInstruction::SourceType::S32,
				SASS::Maxwell::I2IInstruction::Flags::ABS | SASS::Maxwell::I2IInstruction::Flags::NEG
			));

			// XMAD TMP3, TMP4, TMP0, TMP2 ;
			// XMAD.MRG TMP5, TMP4, TMP0.H1, RZ ;
			// XMAD.PSL.CBCC TMP3, TMP4.H1, TMP5.H1, TMP3 ;

			this->AddInstruction(new SASS::Maxwell::XMADInstruction(temp3, temp4, temp0, temp2));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp5, temp4, temp0, SASS::RZ, SASS::Maxwell::XMADInstruction::Mode::MRG, SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp3, temp4, temp5, temp3, SASS::Maxwell::XMADInstruction::Mode::PSL,
				SASS::Maxwell::XMADInstruction::Flags::CBCC | SASS::Maxwell::XMADInstruction::Flags::H1_A | SASS::Maxwell::XMADInstruction::Flags::H1_B
			));

			//     ISETP.GT.U32.AND P, PT, TMP1, TMP3, PT ;
			// @!P IADD TMP3, TMP3, -TMP1 ;
			// @!P IADD32I TMP0, TMP0, 0x1 ;

			this->AddInstruction(new SASS::Maxwell::ISETPInstruction(
				pred, SASS::PT, temp1, temp3, SASS::PT,
				SASS::Maxwell::ISETPInstruction::ComparisonOperator::GT,
				SASS::Maxwell::ISETPInstruction::BooleanOperator::AND,
				SASS::Maxwell::ISETPInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Maxwell::IADDInstruction(temp3, temp3, temp1, SASS::Maxwell::IADDInstruction::Flags::NEG_B), pred, true);
			this->AddInstruction(new SASS::Maxwell::IADD32IInstruction(temp0, temp0, new SASS::I32Immediate(0x1)), pred, true);

			//    ISETP.GE.U32.AND P, PT, TMP3, TMP1, PT ;
			// @P IADD32I TMP0, TMP0, 0x1 ;

			this->AddInstruction(new SASS::Maxwell::ISETPInstruction(
				pred, SASS::PT, temp3, temp1, SASS::PT,
				SASS::Maxwell::ISETPInstruction::ComparisonOperator::GE,
				SASS::Maxwell::ISETPInstruction::BooleanOperator::AND,
				SASS::Maxwell::ISETPInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Maxwell::IADD32IInstruction(temp0, temp0, new SASS::I32Immediate(0x1)), pred);

			//     LOP.XOR TMP3, S1, S2 ;
			//     ISETP.GE.AND P, PT, TMP3, RZ, PT ;
			// @!P IADD TMP0, -TMP0, RZ ;

			this->AddInstruction(new SASS::Maxwell::LOPInstruction(temp3, sourceA, sourceB, SASS::Maxwell::LOPInstruction::BooleanOperator::XOR));
			this->AddInstruction(new SASS::Maxwell::ISETPInstruction(
				pred, SASS::PT, temp3, SASS::RZ, SASS::PT,
				SASS::Maxwell::ISETPInstruction::ComparisonOperator::GE,
				SASS::Maxwell::ISETPInstruction::BooleanOperator::AND
			));
			this->AddInstruction(new SASS::Maxwell::IADDInstruction(temp0, temp0, SASS::RZ, SASS::Maxwell::IADDInstruction::Flags::NEG_A), pred, true);

			//    ISETP.EQ.AND P, PT, S2, RZ, PT ;
			// @P LOP.PASS_B TMP0, RZ, ~S2 ;

			this->AddInstruction(new SASS::Maxwell::ISETPInstruction(
				pred, SASS::PT, sourceB, SASS::RZ, SASS::PT,
				SASS::Maxwell::ISETPInstruction::ComparisonOperator::EQ,
				SASS::Maxwell::ISETPInstruction::BooleanOperator::AND
			));
			this->AddInstruction(new SASS::Maxwell::LOPInstruction(
				temp0, SASS::RZ, sourceB, SASS::Maxwell::LOPInstruction::BooleanOperator::PASS_B, SASS::Maxwell::LOPInstruction::Flags::INV
			), pred);

			// Result

			this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination, temp0));
		}
		else
		{
			Error(instruction, "unsupported type");
		}
	}
	else if constexpr(std::is_same<T, PTX::Float64Type>::value)
	{
		GenerateF64<
			SASS::Maxwell::MOVInstruction, SASS::Maxwell::DMULInstruction,
			SASS::Maxwell::DFMAInstruction, SASS::Maxwell::MUFUInstruction
		>(instruction);
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

template<class T>
void DivideGenerator::GenerateVolta(const PTX::DivideInstruction<T> *instruction)
{
	if constexpr(PTX::is_int_type<T>::value)
	{
		// Generate operands

		RegisterGenerator registerGenerator(this->m_builder);
		CompositeGenerator compositeGenerator(this->m_builder);
		compositeGenerator.SetImmediateSize(32);

		auto destination = registerGenerator.Generate(instruction->GetDestination());
		auto sourceA = registerGenerator.Generate(instruction->GetSourceA());
		auto sourceB = compositeGenerator.Generate(instruction->GetSourceB());

		if constexpr(T::TypeBits == PTX::Bits::Bits32)
		{
			if (auto immediateSourceB = dynamic_cast<SASS::I32Immediate *>(sourceB))
			{
				// Special case for power2 division

				auto value = immediateSourceB->GetValue();
				if (value == Utils::Math::Power2(value))
				{
					auto logValue = Utils::Math::Log2(value);
					immediateSourceB->SetValue(logValue);
					
					auto flags = SASS::Volta::SHFInstruction::Flags::HI;
					auto direction = SASS::Volta::SHFInstruction::Direction::R;

					auto type = SASS::Volta::SHFInstruction::Type::U32;
					if constexpr(PTX::is_signed_int_type<T>::value)
					{
						type = SASS::Volta::SHFInstruction::Type::S32;
					}

					this->AddInstruction(new SASS::Volta::SHFInstruction(
						destination, SASS::RZ, immediateSourceB, sourceA, direction, type, flags
					));
					return;
				}
			}
		}

		// Generate instruction

		if constexpr(std::is_same<T, PTX::UInt32Type>::value)
		{
			auto sourceB = registerGenerator.Generate(instruction->GetSourceB());

			auto [temp1, temp2] = this->m_builder.AllocateTemporaryRegisterPair<PTX::Bits::Bits64>(); // Paired for IMAD.HI
			auto temp1_pair = new SASS::Register(temp1->GetValue(), 2);
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
			// IMAD.MOV TMP1, RZ, RZ, -TMP2 ;
			// IMAD TMP3, S2, TMP1, S1 ;

			this->AddInstruction(new SASS::Volta::MOVInstruction(temp1, SASS::RZ));
			this->AddInstruction(new SASS::Volta::IMADInstruction(temp3, temp3, temp2, SASS::RZ));
			this->AddInstruction(new SASS::Volta::IMADInstruction(
				temp2, temp2, temp3, temp1_pair, SASS::Volta::IMADInstruction::Mode::HI, SASS::Volta::IMADInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Volta::IMADInstruction(
				temp2, temp2, sourceA, SASS::RZ, SASS::Volta::IMADInstruction::Mode::HI, SASS::Volta::IMADInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Volta::IMADInstruction(
				temp3, SASS::RZ, SASS::RZ, temp2, SASS::Volta::IMADInstruction::Mode::Default, SASS::Volta::IMADInstruction::Flags::NEG_C
			));
			this->AddInstruction(new SASS::Volta::IMADInstruction(temp3, sourceB, temp3, sourceA));

			//       ISETP.GE.U32.AND P, PT, TMP3, S2, PT ;
			// @P IADD3 TMP3, -S2, TMP3, RZ ;
			// @P IADD3 TMP2, TMP2, 0x1, RZ ;

			this->AddInstruction(new SASS::Volta::ISETPInstruction(
				pred, SASS::PT, temp3, sourceB, SASS::PT,
				SASS::Volta::ISETPInstruction::ComparisonOperator::GE,
				SASS::Volta::ISETPInstruction::BooleanOperator::AND,
				SASS::Volta::ISETPInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Volta::IADD3Instruction(temp3, sourceB, temp3, SASS::RZ, SASS::Volta::IADD3Instruction::Flags::NEG_A), pred);
			this->AddInstruction(new SASS::Volta::IADD3Instruction(temp2, temp2, new SASS::I32Immediate(0x1), SASS::RZ), pred);

			//    ISETP.GE.U32.AND P, PT, TMP3, S2, PT ;
			// @P IADD3 TMP2, TMP2, 0x1, RZ ;

			this->AddInstruction(new SASS::Volta::ISETPInstruction(
				pred, SASS::PT, temp3, sourceB, SASS::PT,
				SASS::Volta::ISETPInstruction::ComparisonOperator::GE,
				SASS::Volta::ISETPInstruction::BooleanOperator::AND,
				SASS::Volta::ISETPInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Volta::IADD3Instruction(temp2, temp2, new SASS::I32Immediate(0x1), SASS::RZ), pred);

			//    ISETP.EQ.U32.AND P, PT, S2, RZ, PT ;
			// @P LOP3.LUT TMP2, RZ, S2, RZ, 0x33, !PT ;

			this->AddInstruction(new SASS::Volta::ISETPInstruction(
				pred, SASS::PT, SASS::RZ, sourceB, SASS::PT,
				SASS::Volta::ISETPInstruction::ComparisonOperator::EQ,
				SASS::Volta::ISETPInstruction::BooleanOperator::AND,
				SASS::Volta::ISETPInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Volta::LOP3Instruction(
				temp2, SASS::RZ, sourceB, SASS::RZ, new SASS::I8Immediate(0x33), SASS::PT, SASS::Volta::LOP3Instruction::Flags::NOT_E
			), pred);

			// Result

			this->AddInstruction(new SASS::Volta::MOVInstruction(destination, temp2));
		}
		else if constexpr(std::is_same<T, PTX::Int32Type>::value)
		{
			auto sourceB = registerGenerator.Generate(instruction->GetSourceB());

			auto temp1 = this->m_builder.AllocateTemporaryRegister();
			auto temp2 = this->m_builder.AllocateTemporaryRegister();
			auto temp3 = this->m_builder.AllocateTemporaryRegister();
			auto temp4 = this->m_builder.AllocateTemporaryRegister();
			auto [temp5, temp6] = this->m_builder.AllocateTemporaryRegisterPair<PTX::Bits::Bits64>(); // Paired for IMAD.HI
			auto temp5_pair = new SASS::Register(temp5->GetValue(), 2);
			auto temp7 = this->m_builder.AllocateTemporaryRegister();
			auto temp8 = this->m_builder.AllocateTemporaryRegister();
			auto temp9 = this->m_builder.AllocateTemporaryRegister();

			auto pred = this->m_builder.AllocateTemporaryPredicate();

			// IABS TMP1, S2;
			// I2F.RP TMP3, TMP1 ;
			// MUFU.RCP TMP3, TMP3 ;
			// IADD3 TMP5, TMP3, 0xffffffe, RZ ;
			// IABS TMP3, S1 ;
			// F2I.FTZ.U32.TRUNC.NTZ TMP6, TMP5 ;
			// LOP3.LUT TMP4, S1, S2, RZ, 0x3c, !PT ;

			this->AddInstruction(new SASS::Volta::IABSInstruction(temp1, sourceB));
			this->AddInstruction(new SASS::Volta::IABSInstruction(temp9, sourceB));
			this->AddInstruction(new SASS::Volta::I2FInstruction(
				temp3, temp1, SASS::Volta::I2FInstruction::DestinationType::F32, SASS::Volta::I2FInstruction::SourceType::S32,
				SASS::Volta::I2FInstruction::Round::RP
			));
			this->AddInstruction(new SASS::Volta::MUFUInstruction(temp3, temp3, SASS::Volta::MUFUInstruction::Function::RCP));
			this->AddInstruction(new SASS::Volta::IADD3Instruction(temp5, temp3, new SASS::I32Immediate(0xffffffe), SASS::RZ));
			this->AddInstruction(new SASS::Volta::IABSInstruction(temp3, sourceA));
			this->AddInstruction(new SASS::Volta::F2IInstruction(
				temp6, temp5, SASS::Volta::F2IInstruction::DestinationType::U32, SASS::Volta::F2IInstruction::SourceType::F32,
				SASS::Volta::F2IInstruction::Round::TRUNC,
				SASS::Volta::F2IInstruction::Flags::FTZ | SASS::Volta::F2IInstruction::Flags::NTZ
			));
			this->AddInstruction(new SASS::Volta::LOP3Instruction(
				temp4, sourceA, sourceB, SASS::RZ, new SASS::I8Immediate(0x3c), SASS::PT, SASS::Volta::LOP3Instruction::Flags::NOT_E
			));

			// IMAD.MOV.U32 TMP5, RZ, RZ, RZ ;
			// IMAD.MOV TMP7, RZ, RZ, -TMP6 ;
			// IMAD TMP8, TMP7, TMP1, RZ ;
			//
			// IMAD.MOV TMP7, RZ, RZ, -TMP1 ;
			// IMAD.HI.U32 TMP6, TMP6, TMP8, TMP5 ;
			// IMAD.HI.U32 TMP6, TMP6, TMP3, RZ ;
			//
			// IMAD TMP5, TMP6, TMP7, TMP3 ;

			this->AddInstruction(new SASS::Volta::IMADInstruction(
				temp5, SASS::RZ, SASS::RZ, SASS::RZ, SASS::Volta::IMADInstruction::Mode::Default, SASS::Volta::IMADInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Volta::IMADInstruction(
				temp7, SASS::RZ, SASS::RZ, temp6, SASS::Volta::IMADInstruction::Mode::Default, SASS::Volta::IMADInstruction::Flags::NEG_C
			));
			this->AddInstruction(new SASS::Volta::IMADInstruction(temp8, temp7, temp1, SASS::RZ));
			this->AddInstruction(new SASS::Volta::IMADInstruction(
				temp7, SASS::RZ, SASS::RZ, temp9, SASS::Volta::IMADInstruction::Mode::Default, SASS::Volta::IMADInstruction::Flags::NEG_C
			));
			this->AddInstruction(new SASS::Volta::IMADInstruction(
				temp6, temp6, temp8, temp5_pair, SASS::Volta::IMADInstruction::Mode::HI, SASS::Volta::IMADInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Volta::IMADInstruction(
				temp6, temp6, temp3, SASS::RZ, SASS::Volta::IMADInstruction::Mode::HI, SASS::Volta::IMADInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Volta::IMADInstruction(temp5, temp6, temp7, temp3));

			//     ISETP.GT.U32.AND P, PT, TMP1, TMP5, PT ;
			// @!P IMAD.IADD TMP5, TMP5, 0x1, -TMP1 ;
			// @!P IADD3 TMP6, TMP6, 0x1, RZ ;

			this->AddInstruction(new SASS::Volta::ISETPInstruction(
				pred, SASS::PT, temp1, temp5, SASS::PT,
				SASS::Volta::ISETPInstruction::ComparisonOperator::GT,
				SASS::Volta::ISETPInstruction::BooleanOperator::AND,
				SASS::Volta::ISETPInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Volta::IMADInstruction(
				temp5, temp5, new SASS::I32Immediate(0x1), temp1, SASS::Volta::IMADInstruction::Mode::Default,
				SASS::Volta::IMADInstruction::Flags::NEG_C
			), pred, true);
			this->AddInstruction(new SASS::Volta::IADD3Instruction(temp6, temp6, new SASS::I32Immediate(0x1), SASS::RZ), pred, true);

			//    ISETP.GE.U32.AND P, PT, TMP5, TMP1, PT ;
			// @P IADD3 TMP6, TMP6, 0x1, RZ ;

			this->AddInstruction(new SASS::Volta::ISETPInstruction(
				pred, SASS::PT, temp5, temp1, SASS::PT,
				SASS::Volta::ISETPInstruction::ComparisonOperator::GE,
				SASS::Volta::ISETPInstruction::BooleanOperator::AND,
				SASS::Volta::ISETPInstruction::Flags::U32
			));
			this->AddInstruction(new SASS::Volta::IADD3Instruction(temp6, temp6, new SASS::I32Immediate(0x1), SASS::RZ), pred);

			//     ISETP.GE.AND P, PT, TMP4, RZ, PT ;
			// @!P IMAD.MOV TMP6, RZ, RZ, -TMP6 ;

			this->AddInstruction(new SASS::Volta::ISETPInstruction(
				pred, SASS::PT, temp4, SASS::RZ, SASS::PT,
				SASS::Volta::ISETPInstruction::ComparisonOperator::GE,
				SASS::Volta::ISETPInstruction::BooleanOperator::AND
			));
			this->AddInstruction(new SASS::Volta::IMADInstruction(
				temp6, SASS::RZ, SASS::RZ, temp6, SASS::Volta::IMADInstruction::Mode::Default,
				SASS::Volta::IMADInstruction::Flags::NEG_C
			), pred, true);

			//     ISETP.NE.AND P, PT, S2, RZ, PT ;
			// @!P LOP3.LUT TMP6, RZ, S2, RZ, 0x33, !PT ;

			this->AddInstruction(new SASS::Volta::ISETPInstruction(
				pred, SASS::PT, sourceB, SASS::RZ, SASS::PT,
				SASS::Volta::ISETPInstruction::ComparisonOperator::NE,
				SASS::Volta::ISETPInstruction::BooleanOperator::AND
			));
			this->AddInstruction(new SASS::Volta::LOP3Instruction(
				temp6, SASS::RZ, sourceB, SASS::RZ, new SASS::I8Immediate(0x33), SASS::PT, SASS::Volta::LOP3Instruction::Flags::NOT_E
			), pred, true);

			// Result

			this->AddInstruction(new SASS::Volta::MOVInstruction(destination, temp6));
		}
		else
		{
			Error(instruction, "unsupported type");
		}
	}
	else if constexpr(std::is_same<T, PTX::Float64Type>::value)
	{
		GenerateF64<
			SASS::Volta::MOVInstruction, SASS::Volta::DMULInstruction,
			SASS::Volta::DFMAInstruction, SASS::Volta::MUFUInstruction
		>(instruction);
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

template<class MOVInstruction, class DMULInstruction, class DFMAInstruction, class MUFUInstruction>
void DivideGenerator::GenerateF64(const PTX::DivideInstruction<PTX::Float64Type> *instruction)
{
	if (instruction->GetRoundingMode() != PTX::Float64Type::RoundingMode::Nearest)
	{
		Error(instruction, "unsupported rounding mode");
	}

	// Generate operands

	RegisterGenerator registerGenerator(this->m_builder);
	auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());

	auto sourceA = registerGenerator.Generate(instruction->GetSourceA());
	auto sourceB = registerGenerator.Generate(instruction->GetSourceB());
	auto [sourceB_Lo, sourceB_Hi] = registerGenerator.GeneratePair(instruction->GetSourceB());

	// Compute D = S1 / S2

	// MOV32I TMP0_LO, 0x1 ;
	// MUFU.RCP64H TMP0_HI, S2_HI ;
	// DFMA TMP1, S2, -TMP0, c[0x2][0x0] ;
	// DFMA TMP1, TMP1, TMP1, TMP1 ;
	// DFMA TMP1, TMP0, TMP1, TMP0 ;
	// DMUL TMP0, S1, TMP1 ;
	// DFMA TMP2, S2, -TMP0, S1 ;
	// DFMA TMP0, TMP1, TMP2, TMP0 ;

	auto temp0 = this->m_builder.AllocateTemporaryRegister<PTX::Float64Type::TypeBits>();
	auto temp1 = this->m_builder.AllocateTemporaryRegister<PTX::Float64Type::TypeBits>();
	auto temp2 = this->m_builder.AllocateTemporaryRegister<PTX::Float64Type::TypeBits>();

	auto temp0_Lo = new SASS::Register(temp0->GetValue());
	auto temp0_Hi = new SASS::Register(temp0->GetValue() + 1);

	auto temp2_Lo = new SASS::Register(temp2->GetValue());
	auto temp2_Hi = new SASS::Register(temp2->GetValue() + 1);

	// Generate instruction

	auto constant = this->m_builder.AddConstantMemory<double>(1.0);

	this->AddInstruction(new MOVInstruction(temp2_Lo, new SASS::Constant(0x2, constant)));
	this->AddInstruction(new MOVInstruction(temp2_Hi, new SASS::Constant(0x2, constant + 0x4)));

	this->AddInstruction(new MOVInstruction(temp0_Lo, new SASS::I32Immediate(0x1)));
	this->AddInstruction(new MUFUInstruction(temp0_Hi, sourceB_Hi, MUFUInstruction::Function::RCP64H));

	this->AddInstruction(new DFMAInstruction(
		temp1, sourceB, temp0, temp2 /* constant */, DFMAInstruction::Round::RN, DFMAInstruction::Flags::NEG_B
	));
	this->AddInstruction(new DFMAInstruction(temp1, temp1, temp1, temp1));
	this->AddInstruction(new DFMAInstruction(temp1, temp0, temp1, temp0));
	this->AddInstruction(new DMULInstruction(temp0, sourceA, temp1));
	this->AddInstruction(new DFMAInstruction(
		temp2, sourceB, temp0, sourceA, DFMAInstruction::Round::RN, DFMAInstruction::Flags::NEG_B
	));
	this->AddInstruction(new DFMAInstruction(temp0, temp1, temp2, temp0));

	this->AddInstruction(new MOVInstruction(destination_Lo, temp0_Lo));
	this->AddInstruction(new MOVInstruction(destination_Hi, temp0_Hi));
}

}
}
