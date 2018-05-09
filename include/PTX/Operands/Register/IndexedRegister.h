#pragma once

#include <string>

#include "PTX/Operands/Register/Register.h"

namespace PTX {

template<class T, VectorSize V>
class IndexedRegister : public Register<T, Scalar>
{
public:
	IndexedRegister(typename RegisterSpace<T, V>::Element *element, unsigned int index, VectorElement vectorElement) : Register<T, Scalar>(nullptr, index), m_element(element), m_index(index), m_vectorElement(vectorElement) {}

	std::string GetName() const
	{
		return m_element->GetName(m_index) + GetVectorElementName(m_vectorElement);
	}

	virtual VectorElement GetVectorElement() const { return m_vectorElement; }

private:
	typename RegisterSpace<T, V>::Element *m_element = nullptr;
	unsigned int m_index;
	VectorElement m_vectorElement;
};

}
