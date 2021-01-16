#pragma once

#include "PTX/Tree/Operands/Variables/Variable.h"

namespace PTX {

template<class TD, class TS, class S, bool Assert = true>
class VariableAdapter : public Variable<TD, S, Assert>
{
public:
	REQUIRE_TYPE_PARAMS(VariableAdapter,
		REQUIRE_BASE(TD, DataType), REQUIRE_BASE(TS, DataType)
	);
	REQUIRE_SPACE_PARAM(VariableAdapter,
		REQUIRE_BASE(S, StateSpace)
	);

	VariableAdapter(Variable<TS, S> *variable) : Variable<TD, S, Assert>(variable->m_nameSet, variable->m_nameIndex), m_variable(variable) {}

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
	Variable<TS, S> *GetVariable() { return m_variable; }

	// Visitors

	void Accept(OperandVisitor& visitor) override { m_variable->Accept(visitor); }
	void Accept(ConstOperandVisitor& visitor) const override { m_variable->Accept(visitor); }

private:
	Variable<TS, S> *m_variable;
};

}
