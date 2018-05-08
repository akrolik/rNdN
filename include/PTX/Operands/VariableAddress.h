#pragma once

#include "PTX/Operands/Address.h"
#include "PTX/StateSpaces/MemorySpace.h"

namespace PTX {

template<Bits A, class T, VectorSize V = Scalar>
class VariableAddress : public Address<A, T, V>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	VariableAddress(MemorySpace<T, V> *space, int offset = 0) : m_space(space), m_offset(offset) {}

	std::string ToString()
	{
		return "[" + m_space->Name() + "]";
	}

	std::string SpaceName()
	{
		return m_space->SpaceName();
	}

	MemorySpace<T, V> *Space() { return m_space; }
	int Offset() { return m_offset; }

private:
	MemorySpace<T, V> *m_space = nullptr;
	int m_offset = 0;
};

}
