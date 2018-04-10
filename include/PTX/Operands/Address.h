#pragma once

#include "PTX/Operand.h"
#include "PTX/StateSpaces/MemorySpace.h"

namespace PTX {

template<Type T, VectorSize V = Scalar>
class Address : public Operand<T, V>
{
public:
	Address(MemorySpace<T, V> *space, int offset) : m_space(space), m_offset(offset) {}
	Address(MemorySpace<T, V> *space) : m_space(space) {} 

	std::string ToString()
	{
		return "[" + m_space->Name() + "]";
	}

	MemorySpace<T, V> *Space() { return m_space; }
	int Offset() { return m_offset; }

private:
	MemorySpace<T, V> *m_space = nullptr;
	int m_offset = 0;
};

}
