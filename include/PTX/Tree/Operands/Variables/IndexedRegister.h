#pragma once

#include "PTX/Tree/Operands/Adapters/VariableAdapter.h"
#include "PTX/Tree/Operands/Variables/Register.h"

namespace PTX {

template<class T, VectorSize V>
class IndexedRegister : public VariableAdapter<T, VectorType<T, V>, RegisterSpace>
{
public:
	REQUIRE_TYPE_PARAM(IndexedRegister,
		REQUIRE_BASE(T, ScalarType)
	);

	IndexedRegister(const Register<VectorType<T, V>> *variable, VectorElement vectorElement) : VariableAdapter<T, VectorType<T, V>, RegisterSpace>(variable), m_vectorElement(vectorElement) {}

	virtual VectorElement GetVectorElement() const { return m_vectorElement; }

	std::string GetName() const override
	{
		return VariableAdapter<T, VectorType<T, V>, RegisterSpace>::GetName() + GetVectorElementName(m_vectorElement);
	}

	json ToJSON() const override
	{
		json j = VariableAdapter<T, VectorType<T, V>, RegisterSpace>::ToJSON();
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
