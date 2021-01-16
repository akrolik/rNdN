#pragma once

#include "PTX/Tree/Operands/Operand.h"

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Type.h"
#include "PTX/Tree/Declarations/NameSet.h"

namespace PTX {

template<class TD, class TS, class S, bool Assert>
class VariableAdapter;

template<class T, class S, bool Assert = true>
class VariableBase : public virtual Operand
{
	template<class _TD, class _TS, class _S, bool _Assert>
	friend class VariableAdapter;
  
public:
	REQUIRE_TYPE_PARAM(Variable,
		REQUIRE_BASE(T, DataType)
	);
	REQUIRE_SPACE_PARAM(Variable,
		REQUIRE_BASE(S, StateSpace)
	);	

	using VariableType = T;
	using VariableSpace = S;

	// Properties

	virtual std::string GetName() const
	{
		return m_nameSet->GetName(m_nameIndex);
	}

	// Formatting

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

template<class T, class S, bool Assert = true>
class Variable : public VariableBase<T, S, Assert>
{
public:
	using VariableBase<T, S, Assert>::VariableBase;
};

}
