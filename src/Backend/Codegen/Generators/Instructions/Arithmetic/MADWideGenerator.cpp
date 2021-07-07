#include "Backend/Codegen/Generators/Instructions/Arithmetic/MADWideGenerator.h"

#include "Backend/Codegen/Generators/Operands/CompositeGenerator.h"
#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void MADWideGenerator::Generate(const PTX::_MADWideInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void MADWideGenerator::Visit(const PTX::MADWideInstruction<T> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types:
	//   - Int16, Int32
	//   - UInt16, UInt32

	if constexpr(std::is_same<T, PTX::UInt32Type>::value)
	{
		// Generate operands

		RegisterGenerator registerGenerator(this->m_builder);
		CompositeGenerator compositeGenerator(this->m_builder);

		auto [destination_Lo, destination_Hi]  = registerGenerator.GeneratePair(instruction->GetDestination());
		auto sourceA = registerGenerator.Generate(instruction->GetSourceA());
		auto sourceB = compositeGenerator.Generate(instruction->GetSourceB());
		auto [sourceC_Lo, sourceC_Hi] = compositeGenerator.GeneratePair(instruction->GetSourceC());

		// Compute D = (S1 * S2 + S3).lo where S2 is a power of 2
		//
		//    SHR.U32 TMP, S1, 0x1e ;
		//    ISCADD D_LO.CC, S1, S3_LO, 0x2 ;
		//    IADD.X D_HI, TMP, S3_HI

		if (auto immediateSourceB = dynamic_cast<SASS::I32Immediate *>(sourceB))
		{
			auto value = immediateSourceB->GetValue();
			if (value == 1)
			{
				this->AddInstruction(new SASS::IADDInstruction(destination_Lo, sourceA, sourceC_Lo, SASS::IADDInstruction::Flags::CC)); 
				this->AddInstruction(new SASS::IADDInstruction(destination_Hi, SASS::RZ, sourceC_Hi, SASS::IADDInstruction::Flags::X));
			}
			else if (value == Utils::Math::Power2(value))
			{
				auto temp = this->m_builder.AllocateTemporaryRegister();
				auto logValue = Utils::Math::Log2(value);

				this->AddInstruction(new SASS::SHRInstruction(
					temp, sourceA, new SASS::I32Immediate(32 - logValue), SASS::SHRInstruction::Flags::U32
				));
				this->AddInstruction(new SASS::ISCADDInstruction(
					destination_Lo, sourceA, sourceC_Lo, new SASS::I8Immediate(logValue), SASS::ISCADDInstruction::Flags::CC
				));
				this->AddInstruction(new SASS::IADDInstruction(destination_Hi, temp, sourceC_Hi, SASS::IADDInstruction::Flags::X));
			}
			else
			{
				Error(instruction, "requires constant power-2 multiplication value");
			}
		}
		else
		{
			Error(instruction, "requires constant power-2 multiplication value");
		}
	}
	else
	{
		Error(instruction, "unsupported type");
	}
}

}
}
