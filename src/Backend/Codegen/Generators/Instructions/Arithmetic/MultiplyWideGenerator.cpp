#include "Backend/Codegen/Generators/Instructions/Arithmetic/MultiplyWideGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void MultiplyWideGenerator::Generate(const PTX::_MultiplyWideInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void MultiplyWideGenerator::Visit(const PTX::MultiplyWideInstruction<T> *instruction)
{
	// Types:
	//   - Int16, Int32
	//   - UInt16, UInt32
	// Modifiers: --

	// Generate operands

	RegisterGenerator registerGenerator(this->m_builder);
	CompositeGenerator compositeGenerator(this->m_builder);

	auto [destination, destination_Hi] = registerGenerator.Generate(instruction->GetDestination());
	auto [sourceA, sourceA_Hi] = registerGenerator.Generate(instruction->GetSourceA());
	auto [sourceB, sourceB_Hi] = compositeGenerator.Generate(instruction->GetSourceB());

	// Generate instruction

	if constexpr(T::TypeBits == PTX::Bits::Bits32)
	{
		if (auto immediateSourceB = dynamic_cast<SASS::I32Immediate *>(sourceB))
		{
			// Special case for constant multiplications

			auto value = immediateSourceB->GetValue();
			if (value == 0)
			{
				this->AddInstruction(new SASS::MOVInstruction(destination, SASS::RZ));
				this->AddInstruction(new SASS::MOVInstruction(destination_Hi, SASS::RZ));
				return;
			}
			else if (value == 1)
			{
				this->AddInstruction(new SASS::MOVInstruction(destination, sourceA));
				this->AddInstruction(new SASS::MOVInstruction(destination_Hi, SASS::RZ));
				return;
			}
			else if (value == Utils::Math::Power2(value))
			{
				auto temp = this->m_builder.AllocateTemporaryRegister();

				auto logValue = Utils::Math::Log2(value);
				immediateSourceB->SetValue(logValue);

				auto flagsSHR = SASS::SHRInstruction::Flags::None;
				if constexpr(std::is_same<T, PTX::UInt32Type>::value)
				{
					flagsSHR |= SASS::SHRInstruction::Flags::U32;
				}

				this->AddInstruction(new SASS::SHLInstruction(temp, sourceA, immediateSourceB));
				this->AddInstruction(new SASS::SHRInstruction(
					destination_Hi, sourceA, new SASS::I32Immediate(32 - logValue), flagsSHR
				));
				this->AddInstruction(new SASS::MOVInstruction(destination, temp));
				return;
			}

			// All other cases use a complex multiplication, requiring a non-immediate value

			compositeGenerator.SetImmediateValue(false);
			auto [compB, compB_Hi] = compositeGenerator.Generate(instruction->GetSourceB());

			sourceB = compB;
			sourceB_Hi = compB_Hi;
		}
	}

	if constexpr(std::is_same<T, PTX::UInt32Type>::value)
	{
		// Compute {D1, D2} = S1 * S2
		//
		//   XMAD TMP0, S1, S2, RZ ;
		//   XMAD TMP1, S1, S2, RZ ;
		//   XMAD.MRG TMP2, S1, S2.H1, RZ ;
		//   XMAD TMP3, S1, S2.H1, RZ ;
		//   XMAD.CHI TMP0, S1.H1, S2, TMP0 ;
		//   XMAD TMP4, S1.H1, S2.H1, RZ ;
		//   XMAD.PSL.CBCC D1, S1.H1, TMP2.H1, TMP1 ;
		//   IADD3.RS D2, TMP0, TMP3, TMP4 ;

		auto temp0 = this->m_builder.AllocateTemporaryRegister();
		auto temp1 = this->m_builder.AllocateTemporaryRegister();
		auto temp2 = this->m_builder.AllocateTemporaryRegister();
		auto temp3 = this->m_builder.AllocateTemporaryRegister();
		auto temp4 = this->m_builder.AllocateTemporaryRegister();

		this->AddInstruction(new SASS::XMADInstruction(temp0, sourceA, sourceB, SASS::RZ));
		this->AddInstruction(new SASS::XMADInstruction(temp1, sourceA, sourceB, SASS::RZ));
		this->AddInstruction(new SASS::XMADInstruction(
			temp2, sourceA, sourceB, SASS::RZ, SASS::XMADInstruction::Mode::MRG, SASS::XMADInstruction::Flags::H1_B
		));
		this->AddInstruction(new SASS::XMADInstruction(
			temp3, sourceA, sourceB, SASS::RZ, SASS::XMADInstruction::Mode::None, SASS::XMADInstruction::Flags::H1_B
		));
		this->AddInstruction(new SASS::XMADInstruction(
			temp0, sourceA, sourceB, temp0, SASS::XMADInstruction::Mode::CHI, SASS::XMADInstruction::Flags::H1_A
		));
		this->AddInstruction(new SASS::XMADInstruction(
			temp4, sourceA, sourceB, SASS::RZ, SASS::XMADInstruction::Mode::None,
			SASS::XMADInstruction::Flags::H1_A | SASS::XMADInstruction::Flags::H1_B
		));
		this->AddInstruction(new SASS::XMADInstruction(
			destination, sourceA, temp2, temp1, SASS::XMADInstruction::Mode::PSL,
			SASS::XMADInstruction::Flags::CBCC | SASS::XMADInstruction::Flags::H1_A | SASS::XMADInstruction::Flags::H1_B
		));
		this->AddInstruction(new SASS::IADD3Instruction(
			destination_Hi, temp0, temp3, temp4, SASS::IADD3Instruction::Flags::RS
		));
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

}
}
