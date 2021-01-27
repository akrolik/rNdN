#include "Backend/Codegen/Generators/Instructions/Data/PackGenerator.h"

#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

void PackGenerator::Generate(const PTX::_PackInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T, PTX::VectorSize V>
void PackGenerator::Visit(const PTX::PackInstruction<T, V> *instruction)
{
	// Generate destination register

	RegisterGenerator registerGenerator(this->m_builder);
	auto [destination, destinationHi] = registerGenerator.Generate(instruction->GetDestination());

	// Source decomposition, split below

	auto sources = instruction->GetSource()->GetOperands();

	//TODO: Instruction Pack<T, V> types and vectors
	if constexpr(V == PTX::VectorSize::Vector2)
	{
		auto [sourceA, sourceA_Hi] = registerGenerator.Generate(sources.at(0));
		auto [sourceB, sourceB_Hi] = registerGenerator.Generate(sources.at(1));

		if constexpr(std::is_same<T, PTX::Bit16Type>::value)
		{
			auto temp0 = registerGenerator.GenerateTemporary(0);

			this->AddInstruction(new SASS::SHLInstruction(temp0, sourceB, new SASS::I32Immediate(0x8)));
			this->AddInstruction(new SASS::LOPInstruction(destination, sourceA, temp0, SASS::LOPInstruction::BooleanOperator::OR));
		}
		else if constexpr(std::is_same<T, PTX::Bit32Type>::value)
		{
		}
		else if constexpr(std::is_same<T, PTX::Bit64Type>::value)
		{
		}
	}
}

}
}
