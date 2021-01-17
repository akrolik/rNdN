#pragma once

#include "PTX/Tree/Operands/Adapters/VariableAdapter.h"
#include "PTX/Tree/Operands/Variables/Variable.h"

namespace PTX {

DispatchInterface_VectorSpace(IndexedRegister)

template<class T, class S, VectorSize V, bool Assert = true>
class IndexedRegister : DispatchInherit(IndexedRegister), public VariableAdapter<T, VectorType<T, V, Assert>, S, Assert>
{
public:
	REQUIRE_TYPE_PARAM(IndexedRegister,
		REQUIRE_BASE(T, ScalarType)
	);
	REQUIRE_SPACE_PARAM(IndexedRegister,
		REQUIRE_EXACT(S, RegisterSpace, SpecialRegisterSpace)
	);

	IndexedRegister(Variable<VectorType<T, V>, S> *variable, VectorElement vectorElement)
		: VariableAdapter<T, VectorType<T, V, Assert>, S, Assert>(variable), m_vectorElement(vectorElement) {}

	// Properties

	VectorElement GetVectorElement() const { return m_vectorElement; }
	void SetVectorElement(VectorElement vectorElement) { m_vectorElement = vectorElement; }

	std::string GetName() const override
	{
		return VariableAdapter<T, VectorType<T, V, Assert>, S>::GetName() + GetVectorElementName(m_vectorElement);
	}

	// Formatting

	json ToJSON() const override
	{
		json j = VariableAdapter<T, VectorType<T, V>, S>::ToJSON();
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
	DispatchMember_Space(S);
	DispatchMember_Vector(V);

	VectorElement m_vectorElement;
};

template<class T>
using IndexedRegister2 = IndexedRegister<T, RegisterSpace, VectorSize::Vector2>;
template<class T>
using IndexedRegister4 = IndexedRegister<T, RegisterSpace, VectorSize::Vector4>;

template<class T>
using IndexedSpecialRegister2 = IndexedRegister<T, SpecialRegisterSpace, VectorSize::Vector2>;
template<class T>
using IndexedSpecialRegister4 = IndexedRegister<T, SpecialRegisterSpace, VectorSize::Vector4>;

}
