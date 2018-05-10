#pragma once

#include <string>

#include "PTX/Operands/Register/Register.h"

namespace PTX {

template<class T, VectorSize V>
class IndexedRegister : public Register<T, Scalar>
{
public:
	IndexedRegister(typename RegisterSpace<T, V>::Element *element, unsigned int index, VectorElement vectorElement) : Register<T, Scalar>(element->m_structure, index), m_vectorElement(vectorElement) {}

	std::string GetName() const
	{
		return Register<T, Scalar>::GetName() + GetVectorElementName(m_vectorElement);
	}

	virtual VectorElement GetVectorElement() const { return m_vectorElement; }

	friend class RegisterSpace<T, V>;
private:
	IndexedRegister(Structure *structure, unsigned int index, VectorElement vectorElement) : Register<T, Scalar>(structure, index), m_vectorElement(vectorElement) {}

	VectorElement m_vectorElement;
};

}
