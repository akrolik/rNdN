#include "Backend/Codegen/Generators/Instructions/Arithmetic/MultiplyWideGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void MultiplyWideGenerator::Generate(const PTX::_MultiplyWideInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void MultiplyWideGenerator::Visit(const PTX::MultiplyWideInstruction<T> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types:
	//   - Int16, Int32
	//   - UInt16, UInt32
	// Modifiers: --

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<class T>
void MultiplyWideGenerator::GenerateMaxwell(const PTX::MultiplyWideInstruction<T> *instruction)
{
	// Generate operands

	RegisterGenerator registerGenerator(this->m_builder);
	CompositeGenerator compositeGenerator(this->m_builder);

	auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());
	auto sourceA = registerGenerator.Generate(instruction->GetSourceA());
	auto sourceB = compositeGenerator.Generate(instruction->GetSourceB());

	// Generate instruction

	if constexpr(T::TypeBits == PTX::Bits::Bits32)
	{
		if (auto immediateSourceB = dynamic_cast<SASS::I32Immediate *>(sourceB))
		{
			// Special case for constant multiplications

			auto value = immediateSourceB->GetValue();
			if (value == 0)
			{
				this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination_Lo, SASS::RZ));
				this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination_Hi, SASS::RZ));
				return;
			}
			else if (value == 1)
			{
				this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination_Lo, sourceA));
				this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination_Hi, SASS::RZ));
				return;
			}
			else if (value == Utils::Math::Power2(value))
			{
				auto temp = this->m_builder.AllocateTemporaryRegister();

				auto logValue = Utils::Math::Log2(value);
				immediateSourceB->SetValue(logValue);

				auto flagsSHR = SASS::Maxwell::SHRInstruction::Flags::None;
				if constexpr(std::is_same<T, PTX::UInt32Type>::value)
				{
					flagsSHR |= SASS::Maxwell::SHRInstruction::Flags::U32;
				}

				this->AddInstruction(new SASS::Maxwell::SHLInstruction(temp, sourceA, immediateSourceB));
				this->AddInstruction(new SASS::Maxwell::SHRInstruction(
					destination_Hi, sourceA, new SASS::I32Immediate(32 - logValue), flagsSHR
				));
				this->AddInstruction(new SASS::Maxwell::MOVInstruction(destination_Lo, temp));
				return;
			}

			// All other cases use a complex multiplication, requiring a non-immediate value

			compositeGenerator.SetImmediateValue(false);
			sourceB = compositeGenerator.Generate(instruction->GetSourceB());
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

		this->AddInstruction(new SASS::Maxwell::XMADInstruction(temp0, sourceA, sourceB, SASS::RZ));
		this->AddInstruction(new SASS::Maxwell::XMADInstruction(temp1, sourceA, sourceB, SASS::RZ));
		this->AddInstruction(new SASS::Maxwell::XMADInstruction(
			temp2, sourceA, sourceB, SASS::RZ,
			SASS::Maxwell::XMADInstruction::Mode::MRG, SASS::Maxwell::XMADInstruction::Flags::H1_B
		));
		this->AddInstruction(new SASS::Maxwell::XMADInstruction(
			temp3, sourceA, sourceB, SASS::RZ,
			SASS::Maxwell::XMADInstruction::Mode::None, SASS::Maxwell::XMADInstruction::Flags::H1_B
		));
		this->AddInstruction(new SASS::Maxwell::XMADInstruction(
			temp0, sourceA, sourceB, temp0,
			SASS::Maxwell::XMADInstruction::Mode::CHI, SASS::Maxwell::XMADInstruction::Flags::H1_A
		));
		this->AddInstruction(new SASS::Maxwell::XMADInstruction(
			temp4, sourceA, sourceB, SASS::RZ, SASS::Maxwell::XMADInstruction::Mode::None,
			SASS::Maxwell::XMADInstruction::Flags::H1_A | SASS::Maxwell::XMADInstruction::Flags::H1_B
		));
		this->AddInstruction(new SASS::Maxwell::XMADInstruction(
			destination_Lo, sourceA, temp2, temp1, SASS::Maxwell::XMADInstruction::Mode::PSL,
			SASS::Maxwell::XMADInstruction::Flags::CBCC | SASS::Maxwell::XMADInstruction::Flags::H1_A |
			SASS::Maxwell::XMADInstruction::Flags::H1_B
		));
		this->AddInstruction(new SASS::Maxwell::IADD3Instruction(
			destination_Hi, temp0, temp3, temp4, SASS::Maxwell::IADD3Instruction::Flags::RS
		));
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

template<class T>
void MultiplyWideGenerator::GenerateVolta(const PTX::MultiplyWideInstruction<T> *instruction)
{
	Error(instruction, "unsupported architecture");
}

}
}
