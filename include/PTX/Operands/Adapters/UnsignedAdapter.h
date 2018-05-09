#pragma once

#include "PTX/Operands/Register/Register.h"

namespace PTX {

template<Bits B, VectorSize V = Scalar>
class UnsignedAdapter : public Register<UIntType<B>, V>
{
public:
	UnsignedAdapter(Register<IntType<B>> *reg) : Register<UIntType<B>, V>(nullptr, -1), m_register(reg) {}

	std::string Name()
	{
		return m_register->Name();
	}

	std::string ToString()
	{
		return m_register->ToString();
	}

private:
	Register<IntType<B>, V> *m_register = nullptr;
};

}
