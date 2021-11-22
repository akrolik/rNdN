#include "Backend/Codegen/Generators/Instructions/Arithmetic/MADGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void MADGenerator::Generate(const PTX::_MADInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void MADGenerator::Visit(const PTX::MADInstruction<T> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types:
	//   - Int16, Int32, Int64
	//   - UInt16, UInt32, UInt64
	//   - Float32, Float64
	// Modifiers:
	//   - Carry: Int32, Int64, UInt32, UInt64
	//   - Half: Int16, Int32, Int64, UInt16, UInt32, UInt64
	//   - FlushSubnormal: Float32
	//   - Rounding: Float32, Float64
	//   - Saturate: Int32 (Hi only), Float32

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<class T>
void MADGenerator::GenerateMaxwell(const PTX::MADInstruction<T> *instruction)
{
	if constexpr(std::is_same<T, PTX::UInt32Type>::value)
	{
		if (instruction->GetCarryIn() || instruction->GetCarryOut())
		{
			Error(instruction, "unsupported carry modifier");
		}

		if (instruction->GetHalf() == PTX::MADInstruction<T>::Half::Lower)
		{
			// Generate operands

			RegisterGenerator registerGenerator(this->m_builder);
			CompositeGenerator compositeGenerator(this->m_builder);
			compositeGenerator.SetImmediateValue(false);

			auto destination = registerGenerator.Generate(instruction->GetDestination());
			auto sourceA = registerGenerator.Generate(instruction->GetSourceA());
			auto sourceB = compositeGenerator.Generate(instruction->GetSourceB());
			auto sourceC = registerGenerator.Generate(instruction->GetSourceC());

			// Compute D = (S1 * S2 + S3).lo
			//
			//   XMAD TMP0, S1, S2, S3 ;
			//   XMAD.MRG TMP1, S1, S2.H1, RZ ;
			//   XMAD.PSL.CBCC D, S1.H1, TMP1.H1, TMP0 ;

			auto temp0 = this->m_builder.AllocateTemporaryRegister();
			auto temp1 = this->m_builder.AllocateTemporaryRegister();

			this->AddInstruction(new SASS::Maxwell::XMADInstruction(temp0, sourceA, sourceB, sourceC));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				temp1, sourceA, sourceB, SASS::RZ,
				SASS::Maxwell::XMADInstruction::Mode::MRG, SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::Maxwell::XMADInstruction(
				destination, sourceA, temp1, temp0, SASS::Maxwell::XMADInstruction::Mode::PSL,
				SASS::Maxwell::XMADInstruction::Flags::CBCC | SASS::Maxwell::XMADInstruction::Flags::H1_A |
				SASS::Maxwell::XMADInstruction::Flags::H1_B
			));
		}
		else
		{
			Error(instruction, "unsuppoorted half modifier");
		}
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

template<class T>
void MADGenerator::GenerateVolta(const PTX::MADInstruction<T> *instruction)
{
	Error(instruction, "unsupported architecture");
}

}
}
