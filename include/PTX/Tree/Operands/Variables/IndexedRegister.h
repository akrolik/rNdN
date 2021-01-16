#pragma once

#include "PTX/Tree/Operands/Adapters/VariableAdapter.h"
#include "PTX/Tree/Operands/Variables/Register.h"

namespace PTX {

DispatchInterface_Vector(IndexedRegister)

template<class T, VectorSize V, bool Assert = true>
class IndexedRegister : DispatchInherit(IndexedRegister), public VariableAdapter<T, VectorType<T, V, Assert>, RegisterSpace, Assert>
{
public:
	REQUIRE_TYPE_PARAM(IndexedRegister,
		REQUIRE_BASE(T, ScalarType)
	);

	IndexedRegister(Register<VectorType<T, V>> *variable, VectorElement vectorElement)
		: VariableAdapter<T, VectorType<T, V, Assert>, RegisterSpace, Assert>(variable), m_vectorElement(vectorElement) {}

	// Properties

	VectorElement GetVectorElement() const { return m_vectorElement; }
	void SetVectorElement(VectorElement vectorElement) { m_vectorElement = vectorElement; }

	std::string GetName() const override
	{
		return VariableAdapter<T, VectorType<T, V>, RegisterSpace>::GetName() + GetVectorElementName(m_vectorElement);
	}

	// Formatting

	json ToJSON() const override
	{
		json j = VariableAdapter<T, VectorType<T, V>, RegisterSpace>::ToJSON();
		j["kind"] = "PTX::IndexedRegister";
		j["type"] = VectorType<T, V>::Name();
		j["index"] = GetVectorElementName(m_vectorElement);
		return j;
	}

	// Visitors

	void Accept(OperandVisitor& visitor) override { visitor.Visit(static_cast<_IndexedRegister *>(this)); }
	void Accept(ConstOperandVisitor& visitor) const override { visitor.Visit(static_cast<const _IndexedRegister *>(this)); }

protected:
	DispatchMember_Type(T);
	DispatchMember_Vector(V);

	VectorElement m_vectorElement;
};

template<class T>
using IndexedRegister2 = IndexedRegister<T, VectorSize::Vector2>;
template<class T>
using IndexedRegister4 = IndexedRegister<T, VectorSize::Vector4>;

}
