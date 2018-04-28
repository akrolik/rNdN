#pragma once

#include "PTX/Operands/Register.h"

namespace PTX {

template<class T1, class T2, VectorSize V = Scalar>
class ZeroExtendRegister : public Register<T1, V>
{
public:
	ZeroExtendRegister(Register<T2, V> *reg) : Register<T1, V>(nullptr, -1), m_register(reg) {}

	std::string Name()
	{
		return m_register->Name();
	}

	std::string ToString()
	{
		return m_register->ToString();
	}

private:
	Register<T2, V> *m_register = nullptr;
};

}
