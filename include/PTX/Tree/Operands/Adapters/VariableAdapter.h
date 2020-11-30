#pragma once

#include "PTX/Tree/Operands/Variables/AddressableVariable.h"
#include "PTX/Tree/Operands/Variables/Variable.h"

namespace PTX {

template<class TD, class TS, class S>
class VariableAdapter : public Variable<TD, S>
{
public:
	REQUIRE_TYPE_PARAMS(VariableAdapter,
		REQUIRE_BASE(TD, Type), REQUIRE_BASE(TS, Type)
	);
	REQUIRE_SPACE_PARAM(VariableAdapter,
		REQUIRE_BASE(S, StateSpace)
	);

	VariableAdapter(const Variable<TS, S> *variable) : Variable<TD, S>(variable->m_nameSet, variable->m_nameIndex), m_variable(variable) {}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::VariableAdapter";
		j["space"] = S::Name();
		j["destination_type"] = TD::Name();
		j["source_type"] = TS::Name();
		j["variable"] = m_variable->ToJSON();
		return j;
	}

	const Variable<TS, S> *GetVariable() const { return m_variable; }

private:
	const Variable<TS, S> *m_variable;
};

}
