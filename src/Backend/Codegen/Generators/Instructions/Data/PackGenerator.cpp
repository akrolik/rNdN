#include "Backend/Codegen/Generators/Instructions/Data/PackGenerator.h"

#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

#include "Backend/Codegen/Generators/ArchitectureDispatch.h"

namespace Backend {
namespace Codegen {

void PackGenerator::Generate(const PTX::_PackInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T, PTX::VectorSize V>
void PackGenerator::Visit(const PTX::PackInstruction<T, V> *instruction)
{
	// Instruction predicate

	this->SetPredicatedInstruction(instruction);

	// Types:
	//   - Bit16, Bit32, Bit64
	// Vector
	//   - Vector2
	//   - Vector4

	ArchitectureDispatch::Dispatch(*this, instruction);
}

template<class T, PTX::VectorSize V>
void PackGenerator::GenerateMaxwell(const PTX::PackInstruction<T, V> *instruction)
{
	GenerateInstruction<
		SASS::Maxwell::MOVInstruction, SASS::Maxwell::PRMTInstruction
	>(instruction);
}

template<class T, PTX::VectorSize V>
void PackGenerator::GenerateVolta(const PTX::PackInstruction<T, V> *instruction)
{
	GenerateInstruction<
		SASS::Volta::MOVInstruction, SASS::Volta::PRMTInstruction
	>(instruction);
}

template<class MOVInstruction, class PRMTInstruction, class T, PTX::VectorSize V>
void PackGenerator::GenerateInstruction(const PTX::PackInstruction<T, V> *instruction)
{
	// Generate destination register

	RegisterGenerator registerGenerator(this->m_builder);
	auto [destination_Lo, destination_Hi] = registerGenerator.GeneratePair(instruction->GetDestination());

	// Source decomposition, split below

	auto sources = instruction->GetSource()->GetOperands();

	// Generate instruction

	if constexpr(V == PTX::VectorSize::Vector2)
	{
		auto sourceA = registerGenerator.Generate(sources.at(0));
		auto sourceB = registerGenerator.Generate(sources.at(1));

		// Temporary necessary for register reuse

		auto temp = this->m_builder.AllocateTemporaryRegister();

		if constexpr(std::is_same<T, PTX::Bit16Type>::value)
		{
			this->AddInstruction(new PRMTInstruction(destination_Lo, sourceA, new SASS::I32Immediate(0x6540), sourceB));
		}
		else if constexpr(std::is_same<T, PTX::Bit32Type>::value)
		{
			this->AddInstruction(new PRMTInstruction(destination_Lo, sourceA, new SASS::I32Immediate(0x5410), sourceB));
		}
		else if constexpr(std::is_same<T, PTX::Bit64Type>::value)
		{
			if (destination_Lo->GetValue() != sourceB->GetValue())
			{
				this->AddInstruction(new MOVInstruction(destination_Lo, sourceA));
				this->AddInstruction(new MOVInstruction(destination_Hi, sourceB));
			}
			else
			{
				this->AddInstruction(new MOVInstruction(temp, sourceA));
				this->AddInstruction(new MOVInstruction(destination_Hi, sourceB));
				this->AddInstruction(new MOVInstruction(destination_Lo, temp));
			}
		}
	}
	else if constexpr(V == PTX::VectorSize::Vector4)
	{
		Error(instruction, "unsupported vector size");
	}
}

}
}
