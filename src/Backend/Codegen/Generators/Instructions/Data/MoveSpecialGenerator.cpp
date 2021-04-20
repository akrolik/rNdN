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

	this->AddInstruction(new SASS::DEPBARInstruction(
		SASS::DEPBARInstruction::Barrier::SB0, new SASS::I8Immediate(0x0), SASS::DEPBARInstruction::Flags::LE
	));
}

void MoveSpecialGenerator::GenerateS2R(SASS::SpecialRegister *source)
{
	this->AddInstruction(new SASS::S2RInstruction(m_destination, source));
}

void MoveSpecialGenerator::Visit(const PTX::_SpecialRegister *reg)
{
	reg->Dispatch(*this);
}

void MoveSpecialGenerator::Visit(const PTX::_IndexedRegister *reg)
{
	reg->Dispatch(*this);
}

template<class T>
void MoveSpecialGenerator::Visit(const PTX::SpecialRegister<T> *reg)
{
	const auto& name = reg->GetName();
	if constexpr(std::is_same<T, PTX::UInt32Type>::value)
	{
		if (name == PTX::SpecialRegisterName_laneid)
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_LANEID));
		}
		else if (name == PTX::SpecialRegisterName_warpid)
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_VIRTID));
		}
		else if (name == PTX::SpecialRegisterName_nwarpid)
		{
			this->AddInstruction(new SASS::MOVInstruction(m_destination, new SASS::I32Immediate(0x40)));
		}
		else if (name == PTX::SpecialRegisterName_smid)
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_VIRTID));
			this->AddInstruction(new SASS::BFEInstruction(
				m_destination, m_destination, new SASS::I32Immediate(0x914), SASS::BFEInstruction::Flags::U32
			));
		}
		else if (name == PTX::SpecialRegisterName_nsmid)
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_VIRTCFG));
			this->AddInstruction(new SASS::BFEInstruction(
				m_destination, m_destination, new SASS::I32Immediate(0x914), SASS::BFEInstruction::Flags::U32
			));
		}
		else if (name == PTX::SpecialRegisterName_lanemask_eq)
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_EQMASK));
		}
		else if (name == PTX::SpecialRegisterName_lanemask_le)
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_LEMASK));
		}
		else if (name == PTX::SpecialRegisterName_lanemask_lt)
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_LTMASK));
		}
		else if (name == PTX::SpecialRegisterName_lanemask_ge)
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_GEMASK));
		}
		else if (name == PTX::SpecialRegisterName_lanemask_gt)
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_GTMASK));
		}
		else if (name == PTX::SpecialRegisterName_clock)
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_CLOCKLO));
		}
		else if (name == PTX::SpecialRegisterName_clock_hi)
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_CLOCKHI));
		}
		else if (name == std::string(PTX::SpecialRegisterName_pm) + "0")
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_PM0));
		}
		else if (name == std::string(PTX::SpecialRegisterName_pm) + "1")
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_PM1));
		}
		else if (name == std::string(PTX::SpecialRegisterName_pm) + "2")
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_PM2));
		}
		else if (name == std::string(PTX::SpecialRegisterName_pm) + "3")
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_PM3));
		}
		else if (name == std::string(PTX::SpecialRegisterName_pm) + "4")
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_PM4));
		}
		else if (name == std::string(PTX::SpecialRegisterName_pm) + "5")
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_PM5));
		}
		else if (name == std::string(PTX::SpecialRegisterName_pm) + "6")
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_PM6));
		}
		else if (name == std::string(PTX::SpecialRegisterName_pm) + "7")
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_PM7));
		}
		else if (name == PTX::SpecialRegisterName_globaltimer32_lo)
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_GLOBALTIMERLO));
		}
		else if (name == PTX::SpecialRegisterName_globaltimer32_hi)
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_GLOBALTIMERHI));
		}
		else if (name == PTX::SpecialRegisterName_total_smem)
		{
			GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_SMEMSZ));
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
		else if (name == PTX::SpecialRegisterName_clock64)
		{
			//TODO: SR Clock 64-bit

			// MOV R4, RZ ;
			// CS2R R0, SR_CLOCKHI ;
			// CS2R R2, SR_CLOCKLO ;
			// CS2R R3, SR_CLOCKHI ;
			// ISETP.NE.U32.AND P0, PT, R0, R3, PT ;
			// @!P0 BRA 0x70 ;
			// IADD32I R4, R4, 0x1 ;
			// ISETP.GE.U32.AND P0, PT, R4, c[0x0][0x10c], PT ;
			// @!P0 BRA 0x18 ;
		}
		else if (name == PTX::SpecialRegisterName_pm0_64)
		{
			//TODO: SR PM 64-bit

			// CS2R R0, SR_PM_HI0 ;
			// CS2R R2, SR_PM0 ;
			// CS2R R3, SR_PM_HI0 ;
			// ICMP.LT R3, R0, R3, R2 ;
		}
		else if (name == PTX::SpecialRegisterName_pm1_64) {}
		else if (name == PTX::SpecialRegisterName_pm2_64) {}
		else if (name == PTX::SpecialRegisterName_pm3_64) {}
		else if (name == PTX::SpecialRegisterName_pm4_64) {}
		else if (name == PTX::SpecialRegisterName_pm5_64) {}
		else if (name == PTX::SpecialRegisterName_pm6_64) {}
		else if (name == PTX::SpecialRegisterName_pm7_64) {}
		else if (name == PTX::SpecialRegisterName_globaltimer)
		{
			//TODO: SR Global timer 64-bit

			// CS2R R0, SR_GLOBALTIMERHI ;
			// CS2R R2, SR_GLOBALTIMERLO ;
			// CS2R R3, SR_GLOBALTIMERHI ;
			// ISETP.NE.U32.AND P0, PT, R0, R3, PT ;
			// @!P0 BRA 0x70 ;
			// IADD32I R4, R4, 0x1 ;
			// ISETP.GE.U32.AND P0, PT, R4, c[0x0][0x10c], PT ;
			// @!P0 BRA 0x18 ;
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
					GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_TID_X));
					break;
				}
				case PTX::VectorElement::Y:
				{
					GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_TID_Y));
					break;
				}
				case PTX::VectorElement::Z:
				{
					GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_TID_Z));
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
					GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_CTAID_X));
					break;
				}
				case PTX::VectorElement::Y:
				{
					GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_CTAID_Y));
					break;
				}
				case PTX::VectorElement::Z:
				{
					GenerateS2R(new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_CTAID_Z));
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
