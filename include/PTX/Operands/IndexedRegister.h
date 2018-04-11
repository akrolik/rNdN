#pragma once

#include <string>

#include "PTX/Operands/Register.h"

namespace PTX {

template<Type T, VectorSize V>
class IndexedRegister : public Register<T, Scalar>
{
public:
	IndexedRegister(typename RegisterSpace<T, V>::Element *element, unsigned int index, VectorElement vectorElement) : Register<T, Scalar>(nullptr, index), m_element(element), m_index(index), m_vectorElement(vectorElement) {}

	std::string Name()
	{
		return m_element->VariableName(m_index) + VectorElementName();
	}

	std::string ToString()
	{
		return m_element->Name(m_index) + VectorElementName();
	}

	std::string VectorElementName()
	{
		switch (m_vectorElement)
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
	typename RegisterSpace<T, V>::Element *m_element = nullptr;
	unsigned int m_index;
	VectorElement m_vectorElement;
};

}
