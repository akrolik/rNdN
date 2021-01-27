#include "Backend/Codegen/Generators/Instructions/Arithmetic/MADGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void MADGenerator::Generate(const PTX::_MADInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void MADGenerator::Visit(const PTX::MADInstruction<T> *instruction)
{
	// Setup operands

	RegisterGenerator registerGenerator(this->m_builder);
	CompositeGenerator compositeGenerator(this->m_builder);

	const auto [destination, destination_Hi] = registerGenerator.Generate(instruction->GetDestination());
	const auto [sourceA, sourceA_Hi] = registerGenerator.Generate(instruction->GetSourceA());
	const auto [sourceB, sourceB_Hi] = compositeGenerator.Generate(instruction->GetSourceB());
	const auto [sourceC, sourceC_Hi] = registerGenerator.Generate(instruction->GetSourceC());

	// Generate instruction
	//  - Types:
	//  	- Int16, Int32, Int64
	//  	- UInt16, UInt32, UInt64
	//  	- Float32, Float64
	//  - Modifiers:
	//  	- Half: Int16, Int32, Int64, UInt16, UInt32, UInt64
	//  	- Saturate: Int32, Float32
	//  	- Rounding: Float32, Float64
	//  	- FlushSubnormal: Float32
	//  	- Carry: Int32, Int64, UInt32, UInt64

	//TODO: Instruction MAD<T> types/modifiers
	if constexpr(std::is_same<T, PTX::UInt32Type>::value)
	{
		if (instruction->GetHalf() == PTX::MADInstruction<T>::Half::Lower)
		{
			// Compute D = (S1 * S2 + S3).lo
			//
			//   XMAD D, S1, S2, S3 ;
			//   XMAD.MRG TMP0, S1, S2.H1, RZ ;
			//   XMAD.PSL.CBCC D, S1.H1, TMP0.H1, D ;

			auto temp0 = registerGenerator.GenerateTemporary(0);

			this->AddInstruction(new SASS::XMADInstruction(
				destination, sourceA, sourceB, sourceC
			));
			this->AddInstruction(new SASS::XMADInstruction(
				temp0, sourceA, sourceB, SASS::RZ, SASS::XMADInstruction::Mode::MRG, SASS::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::XMADInstruction(
				destination, sourceA, temp0, destination, SASS::XMADInstruction::Mode::PSL,
				SASS::XMADInstruction::Flags::CBCC | SASS::XMADInstruction::Flags::H1_A | SASS::XMADInstruction::Flags::H1_B
			));
		}
	}
}

}
}
