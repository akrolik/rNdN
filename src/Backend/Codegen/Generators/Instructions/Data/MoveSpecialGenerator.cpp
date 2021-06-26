#include "Backend/Codegen/Generators/Instructions/Data/MoveSpecialGenerator.h"

#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void MoveSpecialGenerator::Generate(const PTX::_MoveSpecialInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void MoveSpecialGenerator::Visit(const PTX::MoveSpecialInstruction<T> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types:
	//   - Bit32
	//   - UInt32, UInt64
	//   - Vector2<UInt32>, Vector4<UInt32>
	// Modifies: --

	// Generate destination

	RegisterGenerator registerGenerator(this->m_builder);
	auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());

	m_destination = destination_Lo;
	m_destinationHi = destination_Hi;

	// Generate instruction depending on source

	instruction->GetSource()->Accept(*this);
}

void MoveSpecialGenerator::GenerateS2R(SASS::SpecialRegister::Kind special)
{
	this->AddInstruction(new SASS::S2RInstruction(m_destination, new SASS::SpecialRegister(special)));
}

void MoveSpecialGenerator::GeneratePM64(SASS::SpecialRegister::Kind specialLo, SASS::SpecialRegister::Kind specialHi)
{
	// CS2R R0, SR_PM_HI0 ;
	// CS2R R2, SR_PM0 ;
	// CS2R R3, SR_PM_HI0 ;
	// ICMP.LT R3, R0, R3, R2 ;

	auto temp = this->m_builder.AllocateTemporaryRegister();

	this->AddInstruction(new SASS::CS2RInstruction(temp, new SASS::SpecialRegister(specialHi)));
	this->AddInstruction(new SASS::CS2RInstruction(m_destination, new SASS::SpecialRegister(specialLo)));
	this->AddInstruction(new SASS::CS2RInstruction(m_destinationHi, new SASS::SpecialRegister(specialHi)));
	this->AddInstruction(new SASS::ICMPInstruction(
		m_destinationHi, temp, m_destinationHi, m_destination, SASS::ICMPInstruction::ComparisonOperator::LT
	));
}

bool MoveSpecialGenerator::Visit(const PTX::_SpecialRegister *reg)
{
	reg->Dispatch(*this);
	return false;
}

bool MoveSpecialGenerator::Visit(const PTX::_IndexedRegister *reg)
{
	reg->Dispatch(*this);
	return false;
}

template<class T>
void MoveSpecialGenerator::Visit(const PTX::SpecialRegister<T> *reg)
{
	const auto& name = reg->GetName();
	if constexpr(std::is_same<T, PTX::UInt32Type>::value)
	{
		if (name == PTX::SpecialRegisterName_laneid)
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_LANEID);
		}
		else if (name == PTX::SpecialRegisterName_warpid)
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_VIRTID);
		}
		else if (name == PTX::SpecialRegisterName_nwarpid)
		{
			this->AddInstruction(new SASS::MOVInstruction(m_destination, new SASS::I32Immediate(0x40)));
		}
		else if (name == PTX::SpecialRegisterName_smid)
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_VIRTID);
			this->AddInstruction(new SASS::BFEInstruction(
				m_destination, m_destination, new SASS::I32Immediate(0x914), SASS::BFEInstruction::Flags::U32
			));
		}
		else if (name == PTX::SpecialRegisterName_nsmid)
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_VIRTCFG);
			this->AddInstruction(new SASS::BFEInstruction(
				m_destination, m_destination, new SASS::I32Immediate(0x914), SASS::BFEInstruction::Flags::U32
			));
		}
		else if (name == PTX::SpecialRegisterName_lanemask_eq)
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_EQMASK);
		}
		else if (name == PTX::SpecialRegisterName_lanemask_le)
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_LEMASK);
		}
		else if (name == PTX::SpecialRegisterName_lanemask_lt)
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_LTMASK);
		}
		else if (name == PTX::SpecialRegisterName_lanemask_ge)
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_GEMASK);
		}
		else if (name == PTX::SpecialRegisterName_lanemask_gt)
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_GTMASK);
		}
		else if (name == PTX::SpecialRegisterName_clock)
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_CLOCKLO);
		}
		else if (name == PTX::SpecialRegisterName_clock_hi)
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_CLOCKHI);
		}
		else if (name == std::string(PTX::SpecialRegisterName_pm) + "0")
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_PM0);
		}
		else if (name == std::string(PTX::SpecialRegisterName_pm) + "1")
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_PM1);
		}
		else if (name == std::string(PTX::SpecialRegisterName_pm) + "2")
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_PM2);
		}
		else if (name == std::string(PTX::SpecialRegisterName_pm) + "3")
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_PM3);
		}
		else if (name == std::string(PTX::SpecialRegisterName_pm) + "4")
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_PM4);
		}
		else if (name == std::string(PTX::SpecialRegisterName_pm) + "5")
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_PM5);
		}
		else if (name == std::string(PTX::SpecialRegisterName_pm) + "6")
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_PM6);
		}
		else if (name == std::string(PTX::SpecialRegisterName_pm) + "7")
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_PM7);
		}
		else if (name == PTX::SpecialRegisterName_globaltimer32_lo)
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_GLOBALTIMERLO);
		}
		else if (name == PTX::SpecialRegisterName_globaltimer32_hi)
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_GLOBALTIMERHI);
		}
		else if (name == PTX::SpecialRegisterName_total_smem)
		{
			GenerateS2R(SASS::SpecialRegister::Kind::SR_SMEMSZ);
		}
		else if (name == PTX::SpecialRegisterName_dynamic_smem)
		{
			this->AddInstruction(new SASS::MOVInstruction(m_destination, new SASS::Constant(0x0, 0xfc)));
		}
	}
	else if constexpr(std::is_same<T, PTX::UInt64Type>::value)
	{
		if (name == PTX::SpecialRegisterName_gridid)
		{
			this->AddInstruction(new SASS::MOVInstruction(m_destination, new SASS::Constant(0x0, 0x28)));
			this->AddInstruction(new SASS::MOVInstruction(m_destinationHi, new SASS::Constant(0x0, 0x2c)));
		}
		else if (name == PTX::SpecialRegisterName_clock64 || name == PTX::SpecialRegisterName_globaltimer)
		{
			//      MOV R4, RZ ;
			// .L0:
			//      CS2R R0, SR_CLOCKHI/SR_GLOBALTIMERHI ;
			//      CS2R R2, SR_CLOCKLO/SR_GLOBALTIMERLO ;
			//      CS2R R3, SR_CLOCKHI/SR_GLOBALTIMERHI ;
			//      ISETP.NE.U32.AND P0, PT, R0, R3, PT ;
			// @!P0 BRA L1 ;
			//      IADD32I R4, R4, 0x1 ;
			//      ISETP.GE.U32.AND P0, PT, R4, c[0x0][0x10c], PT ;
			// @!P0 BRA L0 ;
			//
			// .L1:

			auto temp0 = this->m_builder.AllocateTemporaryRegister();
			auto temp1 = this->m_builder.AllocateTemporaryRegister();
			auto predicate = this->m_builder.AllocateTemporaryPredicate();

			auto currentBlock = this->m_builder.GetCurrentBlock();
			auto currentName = currentBlock->GetName();

			auto label0 = this->m_builder.UniqueIdentifier(currentName + "_CLOCK_START");
			auto label1 = this->m_builder.UniqueIdentifier(currentName + "_CLOCK_MID");
			auto label2 = this->m_builder.UniqueIdentifier(currentName + "_CLOCK_END");

			this->AddInstruction(new SASS::MOVInstruction(temp0, SASS::RZ));

			this->m_builder.CloseBasicBlock();
			this->m_builder.CreateBasicBlock(label0);

			auto specialLo = (name == PTX::SpecialRegisterName_clock64) ?
				SASS::SpecialRegister::Kind::SR_CLOCKLO : SASS::SpecialRegister::Kind::SR_GLOBALTIMERLO;
			auto specialHi = (name == PTX::SpecialRegisterName_clock64) ?
				SASS::SpecialRegister::Kind::SR_CLOCKHI : SASS::SpecialRegister::Kind::SR_GLOBALTIMERHI;

			this->AddInstruction(new SASS::CS2RInstruction(temp1, new SASS::SpecialRegister(specialHi)));
			this->AddInstruction(new SASS::CS2RInstruction(m_destination, new SASS::SpecialRegister(specialLo)));
			this->AddInstruction(new SASS::CS2RInstruction(m_destinationHi, new SASS::SpecialRegister(specialHi)));
			this->AddInstruction(new SASS::ISETPInstruction(
				predicate, SASS::PT, temp1, m_destinationHi, SASS::PT,
				SASS::ISETPInstruction::ComparisonOperator::NE, SASS::ISETPInstruction::BooleanOperator::AND,
				SASS::ISETPInstruction::Flags::U32
			));

			this->AddInstruction(new SASS::BRAInstruction(label1), predicate, true);

			this->m_builder.CloseBasicBlock();
			this->m_builder.CreateBasicBlock(label1);

			this->AddInstruction(new SASS::IADD32IInstruction(temp0, temp0, new SASS::I32Immediate(0x1)));
			this->AddInstruction(new SASS::ISETPInstruction(
				predicate, SASS::PT, temp1, new SASS::Constant(0x0, 0x10c), SASS::PT,
				SASS::ISETPInstruction::ComparisonOperator::GE, SASS::ISETPInstruction::BooleanOperator::AND,
				SASS::ISETPInstruction::Flags::U32
			));

			this->AddInstruction(new SASS::BRAInstruction(label0), predicate, true);

			this->m_builder.CloseBasicBlock();
			this->m_builder.CreateBasicBlock(label2);
		}
		else if (name == PTX::SpecialRegisterName_pm0_64)
		{
			GeneratePM64(SASS::SpecialRegister::Kind::SR_PM0, SASS::SpecialRegister::Kind::SR_PM_HI0);
		}
		else if (name == PTX::SpecialRegisterName_pm1_64)
		{
			GeneratePM64(SASS::SpecialRegister::Kind::SR_PM1, SASS::SpecialRegister::Kind::SR_PM_HI1);
		}
		else if (name == PTX::SpecialRegisterName_pm2_64)
		{
			GeneratePM64(SASS::SpecialRegister::Kind::SR_PM2, SASS::SpecialRegister::Kind::SR_PM_HI2);
		}
		else if (name == PTX::SpecialRegisterName_pm3_64)
		{
			GeneratePM64(SASS::SpecialRegister::Kind::SR_PM3, SASS::SpecialRegister::Kind::SR_PM_HI3);
		}
		else if (name == PTX::SpecialRegisterName_pm4_64)
		{
			GeneratePM64(SASS::SpecialRegister::Kind::SR_PM4, SASS::SpecialRegister::Kind::SR_PM_HI4);
		}
		else if (name == PTX::SpecialRegisterName_pm5_64)
		{
			GeneratePM64(SASS::SpecialRegister::Kind::SR_PM5, SASS::SpecialRegister::Kind::SR_PM_HI5);
		}
		else if (name == PTX::SpecialRegisterName_pm6_64)
		{
			GeneratePM64(SASS::SpecialRegister::Kind::SR_PM6, SASS::SpecialRegister::Kind::SR_PM_HI6);
		}
		else if (name == PTX::SpecialRegisterName_pm7_64)
		{
			GeneratePM64(SASS::SpecialRegister::Kind::SR_PM7, SASS::SpecialRegister::Kind::SR_PM_HI7);
		}
	}
	else if constexpr(std::is_same<T, PTX::Bit32Type>::value)
	{
		if (name.find(PTX::SpecialRegisterName_envreg) == 0)
		{
			this->AddInstruction(new SASS::MOVInstruction(m_destination, SASS::RZ));
		}
	}
}

template<class T, class S, PTX::VectorSize V>
void MoveSpecialGenerator::Visit(const PTX::IndexedRegister<T, S, V> *reg)
{
	if constexpr(std::is_same<PTX::IndexedRegister<T, S, V>, PTX::IndexedRegister<PTX::UInt32Type, PTX::SpecialRegisterSpace, PTX::VectorSize::Vector4>>::value)
	{
		const auto& name = reg->GetVariable()->GetName();
		if (name == PTX::SpecialRegisterName_tid)
		{
			switch (reg->GetVectorElement())
			{
				case PTX::VectorElement::X:
				{
					GenerateS2R(SASS::SpecialRegister::Kind::SR_TID_X);
					break;
				}
				case PTX::VectorElement::Y:
				{
					GenerateS2R(SASS::SpecialRegister::Kind::SR_TID_Y);
					break;
				}
				case PTX::VectorElement::Z:
				{
					GenerateS2R(SASS::SpecialRegister::Kind::SR_TID_Z);
					break;
				}
			}
		}
		else if (name == PTX::SpecialRegisterName_ntid)
		{
			// %ntid.x: c[0x0][0x8]
			// %ntid.y: c[0x0][0xc]
			// %ntid.z: c[0x0][0x10]

			switch (reg->GetVectorElement())
			{
				case PTX::VectorElement::X:
				{
					this->AddInstruction(new SASS::MOVInstruction(m_destination, new SASS::Constant(0x0, 0x8)));
					break;
				}
				case PTX::VectorElement::Y:
				{
					this->AddInstruction(new SASS::MOVInstruction(m_destination, new SASS::Constant(0x0, 0xc)));
					break;
				}
				case PTX::VectorElement::Z:
				{
					this->AddInstruction(new SASS::MOVInstruction(m_destination, new SASS::Constant(0x0, 0x10)));
					break;
				}
			}
		}
		else if (name == PTX::SpecialRegisterName_ctaid)
		{
			switch (reg->GetVectorElement())
			{
				case PTX::VectorElement::X:
				{
					GenerateS2R(SASS::SpecialRegister::Kind::SR_CTAID_X);
					break;
				}
				case PTX::VectorElement::Y:
				{
					GenerateS2R(SASS::SpecialRegister::Kind::SR_CTAID_Y);
					break;
				}
				case PTX::VectorElement::Z:
				{
					GenerateS2R(SASS::SpecialRegister::Kind::SR_CTAID_Z);
					break;
				}
			}
		}
		else if (name == PTX::SpecialRegisterName_nctaid)
		{
			// %nctaid.x: c[0x0][0x14]
			// %nctaid.y: c[0x0][0x18]
			// %nctaid.z: c[0x0][0x1c]

			switch (reg->GetVectorElement())
			{
				case PTX::VectorElement::X:
				{
					this->AddInstruction(new SASS::MOVInstruction(m_destination, new SASS::Constant(0x0, 0x14)));
					break;
				}
				case PTX::VectorElement::Y:
				{
					this->AddInstruction(new SASS::MOVInstruction(m_destination, new SASS::Constant(0x0, 0x18)));
					break;
				}
				case PTX::VectorElement::Z:
				{
					this->AddInstruction(new SASS::MOVInstruction(m_destination, new SASS::Constant(0x0, 0x1c)));
					break;
				}
			}
		}
	}
}

}
}
