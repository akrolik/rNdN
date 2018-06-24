#pragma once

#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T, VectorSize V>
class IndexedRegister : public Register<T>
{
	REQUIRE_BASE_TYPE(IndexedRegister, ScalarType);
public:
	IndexedRegister(const Register<VectorType<T, V>> *variable, VectorElement vectorElement) : Register<T>(variable->GetName()), m_vectorElement(vectorElement) {}

	virtual VectorElement GetVectorElement() const { return m_vectorElement; }

	std::string GetName() const override
	{
		return Register<T>::GetName() + GetVectorElementName(m_vectorElement);
	}

	json ToJSON() const override
	{
		json j = Register<T>::ToJSON();
		j["kind"] = "PTX::IndexedRegister";
		j["type"] = VectorType<T, V>::Name();
		j["index"] = GetVectorElementName(m_vectorElement);
		return j;
	}

protected:
	VectorElement m_vectorElement;
};

template<class T>
using IndexedRegister2 = IndexedRegister<T, VectorSize::Vector2>;
template<class T>
using IndexedRegister4 = IndexedRegister<T, VectorSize::Vector4>;

}
