#pragma once

#include "PTX/Operands/Register/Register.h"

namespace PTX {

template<Bits B, VectorSize V = Scalar>
class SignedAdapter : public Register<IntType<B>, V>
{
public:
	SignedAdapter(Register<UIntType<B>> *reg) : Register<IntType<B>, V>(nullptr, -1), m_register(reg) {}

	std::string Name()
	{
		return m_register->Name();
	}

	std::string ToString()
	{
		return m_register->ToString();
	}

private:
	Register<UIntType<B>, V> *m_register = nullptr;
};

}
