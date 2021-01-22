#include "Backend/Codegen/Generators/Instructions/Data/MoveSpecialGenerator.h"

#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"
#include "Backend/Codegen/Generators/Operands/SpecialRegisterGenerator.h"

namespace Backend {
namespace Codegen {

void MoveSpecialGenerator::Generate(const PTX::_MoveSpecialInstruction *instruction)
{
	instruction->Dispatch(*this);
}

template<class T>
void MoveSpecialGenerator::Visit(const PTX::MoveSpecialInstruction<T> *instruction)
{
	RegisterGenerator registerGenerator(this->m_builder);
	auto destination = registerGenerator.Generate(instruction->GetDestination());

	SpecialRegisterGenerator specialGenerator(this->m_builder);
	auto source = specialGenerator.Generate(instruction->GetSource());

	this->AddInstruction(new SASS::S2RInstruction(destination, source));
}

}
}
