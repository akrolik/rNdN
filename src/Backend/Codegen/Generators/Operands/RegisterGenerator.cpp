#include "Backend/Codegen/Generators/Operands/RegisterGenerator.h"

namespace Backend {
namespace Codegen {

SASS::Register *RegisterGenerator::Generate(const PTX::_Register *reg)
{
	reg->Dispatch(*this);
	if (m_register == nullptr)
	{
		//TODO:
		Error("Unable to generate register for operand 'TODO'");
	}
	return m_register;
}

template<class T>
void RegisterGenerator::Visit(const PTX::Register<T> *reg)
{
	const auto& allocations = this->m_builder.GetRegisterAllocation();
	const auto& [registerAllocation, registerRange] = allocations->GetRegister(reg->GetName());
	m_register = new SASS::Register(registerAllocation);
}

}
}
