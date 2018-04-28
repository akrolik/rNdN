#pragma once

#include "PTX/Operand.h"
#include "PTX/StateSpaces/MemorySpace.h"

namespace PTX {

template<Bits A, class T, VectorSize V = Scalar>
class RegisterAddress : public Address<A, T, V>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	RegisterAddress(Register<UIntType<A>> *reg, int offset = 0) : m_register(reg), m_offset(offset) {}

	std::string ToString()
	{
		return "[" + m_register->Name() + "]";
	}

	std::string SpaceName()
	{
		return ".<unknown>";
	}

	Register<UIntType<A>> *GetRegister() { return m_register; }
	int Offset() { return m_offset; }

private:
	Register<UIntType<A>> *m_register = nullptr;
	int m_offset = 0;
};

}
