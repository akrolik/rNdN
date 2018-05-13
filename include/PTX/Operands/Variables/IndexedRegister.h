#pragma once

#include "PTX/Operands/Variables/Register.h"

#include "PTX/StateSpaces/SpaceAdapter.h"

namespace PTX {

template<class T, VectorSize V>
class IndexedRegister : public Register<T>
{
public:
	IndexedRegister(Register<VectorType<T, V>> *variable, VectorElement vectorElement) : Register<T>(variable->GetName(), new RegisterSpaceAdapter<T, VectorType<T, V>>(variable->GetStateSpace())), m_vectorElement(vectorElement) {}

	virtual VectorElement GetVectorElement() const { return m_vectorElement; }

	std::string ToString() const
	{
		return Register<T>::ToString() + GetVectorElementName(m_vectorElement);
	}

protected:
	VectorElement m_vectorElement;
};

}
