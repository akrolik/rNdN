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
	// Types:
	//   - Int16, Int32, Int64
	//   - UInt16, UInt32, UInt64
	//   - Float32, Float64
	// Modifiers:
	//   - Carry: Int32, Int64, UInt32, UInt64
	//   - Half: Int16, Int32, Int64, UInt16, UInt32, UInt64
	//   - FlushSubnormal: Float32
	//   - Rounding: Float32, Float64
	//   - Saturate: Int32, Float32

	// Generate operands

	RegisterGenerator registerGenerator(this->m_builder);
	CompositeGenerator compositeGenerator(this->m_builder);

	auto [destination, destination_Hi] = registerGenerator.Generate(instruction->GetDestination());
	auto [sourceA, sourceA_Hi] = registerGenerator.Generate(instruction->GetSourceA());
	auto [sourceB, sourceB_Hi] = compositeGenerator.Generate(instruction->GetSourceB());
	auto [sourceC, sourceC_Hi] = registerGenerator.Generate(instruction->GetSourceC());

	// Generate instruction

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

			auto temp = this->m_builder.AllocateTemporaryRegister();

			this->AddInstruction(new SASS::XMADInstruction(
				destination, sourceA, sourceB, sourceC
			));
			this->AddInstruction(new SASS::XMADInstruction(
				temp, sourceA, sourceB, SASS::RZ, SASS::XMADInstruction::Mode::MRG, SASS::XMADInstruction::Flags::H1_B
			));
			this->AddInstruction(new SASS::XMADInstruction(
				destination, sourceA, temp, destination, SASS::XMADInstruction::Mode::PSL,
				SASS::XMADInstruction::Flags::CBCC | SASS::XMADInstruction::Flags::H1_A | SASS::XMADInstruction::Flags::H1_B
			));
		}
	}
}

}
}
