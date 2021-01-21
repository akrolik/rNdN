#include "Backend/Codegen/Generators/Operands/SpecialRegisterGenerator.h"

namespace Backend {
namespace Codegen {

SASS::SpecialRegister *SpecialRegisterGenerator::Generate(const PTX::_SpecialRegister *reg)
{
	reg->Dispatch(*this);
	if (m_register == nullptr)
	{
		//TODO:
		Error("Unable to generate special register for operand 'TODO'");
	}
	return m_register;
}

template<class T>
void SpecialRegisterGenerator::Visit(const PTX::SpecialRegister<T> *reg)
{
	//TODO: All special regs
	if constexpr(std::is_same<T, PTX::UInt32Type>::value)
	{
		const auto& name = reg->GetName();
		if (name == "%tid.x")
		{
			m_register = new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_TID_X);
		}
		else if (name == "%ctaid.x")
		{
			m_register = new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_CTAID_X);
		}
		else if (name == "%ntid.x")
		{
			m_register = new SASS::SpecialRegister(SASS::SpecialRegister::Kind::SR_NTID_X);
		}
	}
	else if constexpr(std::is_same<T, PTX::UInt64Type>::value)
	{
	}
}

}
}
