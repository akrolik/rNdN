#pragma once

#include <string>

#include "PTX/Operands/Register.h"

namespace PTX {

enum Element {
	X,
	Y,
	Z
};

template<Type T, VectorSize V>
class IndexedRegister : public Register<T, Scalar>
{
public:
	IndexedRegister(RegisterSpace<T, V> *space, unsigned int index, Element e) : Register<T, Scalar>(nullptr, index), m_space(space), m_element(e) {}

	std::string ToString()
	{
		return "%" + m_space->Name(this->m_index) + ElementName();
	}

	std::string Name()
	{
		return m_space->Name(this->m_index) + ElementName();
	}

	std::string ElementName()
	{
		switch (m_element)
		{
			case X:
				return ".x";
			case Y:
				return ".y";
			case Z:
				return ".z";
		}
	}

private:
	RegisterSpace<T, V> *m_space = nullptr;
	Element m_element;
};

}
