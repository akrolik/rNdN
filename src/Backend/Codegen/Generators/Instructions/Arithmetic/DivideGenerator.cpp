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
	RegisterGenerator registerGenerator(this->m_builder);
	CompositeGenerator compositeGenerator(this->m_builder);

	if constexpr(PTX::is_int_type<T>::value)
	{
		// Generate operands

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
		if (instruction->GetRoundingMode() != T::RoundingMode::Nearest)
		{
			Error(instruction, "unsupported rounding mode");
		}

		// Generate operands

		auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());

		auto sourceA = registerGenerator.Generate(instruction->GetSourceA());
		auto sourceB = registerGenerator.Generate(instruction->GetSourceB());
		auto [sourceB_Lo, sourceB_Hi] = registerGenerator.GeneratePair(instruction->GetSourceB());

		// Comptue D = S1 / S2

		// MOV32I TMP0_LO, 0x1 ;
		// MUFU.RCP64H TMP0_HI, S2_HI ;
		// DFMA TMP1, S2, -TMP0, c[0x2][0x0] ;
		// DFMA TMP1, TMP1, TMP1, TMP1 ;
		// DFMA TMP1, TMP0, TMP1, TMP0 ;
		// DMUL TMP0, S1, TMP1 ;
		// DFMA TMP2, S2, -TMP0, S1 ;
		// DFMA TMP0, TMP1, TMP2, TMP0 ;

		auto temp0 = this->m_builder.AllocateTemporaryRegister<T::TypeBits>();
		auto temp1 = this->m_builder.AllocateTemporaryRegister<T::TypeBits>();
		auto temp2 = this->m_builder.AllocateTemporaryRegister<T::TypeBits>();

		auto temp0_Lo = new SASS::Register(temp0->GetValue());
		auto temp0_Hi = new SASS::Register(temp0->GetValue() + 1);

		auto temp2_Lo = new SASS::Register(temp2->GetValue());
		auto temp2_Hi = new SASS::Register(temp2->GetValue() + 1);

		// Generate instruction

		auto constant = this->m_builder.AddConstantMemory<double>(1.0);

		this->AddInstruction(new SASS::Maxwell::MOVInstruction(temp2_Lo, new SASS::Constant(0x2, constant)));
		this->AddInstruction(new SASS::Maxwell::MOVInstruction(temp2_Hi, new SASS::Constant(0x2, constant + 0x4)));

		this->AddInstruction(new SASS::Maxwell::MOV32IInstruction(temp0_Lo, new SASS::I32Immediate(0x1)));
		this->AddInstruction(new SASS::Maxwell::MUFUInstruction(temp0_Hi, sourceB_Hi, SASS::Maxwell::MUFUInstruction::Function::RCP64H));

		this->AddInstruction(new SASS::Maxwell::DFMAInstruction(
			temp1, sourceB, temp0, temp2 /* constant */, SASS::Maxwell::DFMAInstruction::Round::RN, SASS::Maxwell::DFMAInstruction::Flags::NEG_B
		));
		this->AddInstruction(new SASS::Maxwell::DFMAInstruction(temp1, temp1, temp1, temp1));
		this->AddInstruction(new SASS::Maxwell::DFMAInstruction(temp1, temp0, temp1, temp0));
		this->AddInstruction(new SASS::Maxwell::DMULInstruction(temp0, sourceA, temp1));
		this->AddInstruction(new SASS::Maxwell::DFMAInstruction(
			temp2, sourceB, temp0, sourceA, SASS::Maxwell::DFMAInstruction::Round::RN, SASS::Maxwell::DFMAInstruction::Flags::NEG_B
		));
		this->AddInstruction(new SASS::Maxwell::DFMAInstruction(temp0, temp1, temp2, temp0));

		this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination_Lo, temp0_Lo));
		this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination_Hi, temp0_Hi));
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

template<class T>
void DivideGenerator::GenerateVolta(const PTX::DivideInstruction<T> *instruction)
{
	Error(instruction, "unsupported architecture");
}

}
}
