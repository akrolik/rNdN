#pragma once

#include "PTX/Operands/Operand.h"

#include "PTX/StateSpace.h"
#include "PTX/Type.h"

namespace PTX {

template<class T, class S>
class TypedVariableDeclaration;

template<class TD, class TS, class S>
class VariableAdapter;

template<class T, class S>
class VariableBase : public virtual Operand
{
	friend class TypedVariableDeclaration<T, S>;

	template<class T1, class T2, class T3>
	friend class VariableAdapter;

	REQUIRE_TYPE_PARAM(Variable,
		REQUIRE_BASE(T, Type)
	);
	REQUIRE_SPACE_PARAM(Variable,
		REQUIRE_BASE(S, StateSpace)
	);	
public:
	using VariableType = T;
	using VariableSpace = S;

	virtual std::string GetName() const
	{
		return m_nameSet->GetName(m_nameIndex);
	}

	std::string ToString() const override
	{
		return GetName();
	}

	json ToJSON () const override
	{
		json j;
		j["kind"] = "PTX::Variable";
		j["name"] = GetName();
		j["type"] = T::Name();
		j["space"] = S::Name();
		return j;
	}

protected:
	VariableBase(const NameSet *nameSet, unsigned int nameIndex) : m_nameSet(nameSet), m_nameIndex(nameIndex) {}

	const NameSet *m_nameSet = nullptr;
	unsigned int m_nameIndex = 0;
};

template<class T, class S, typename Enabled = void>
class Variable : public VariableBase<T, S>
{
public:
	using VariableBase<T, S>::VariableBase;
};

}
