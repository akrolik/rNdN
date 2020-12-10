#pragma once

#include "PTX/Tree/Statements/InstructionStatement.h"

#include <utility>

#include "PTX/Tree/Type.h"
#include "PTX/Tree/Operands/Variables/Register.h"

namespace PTX {

class PredicatedInstruction : public InstructionStatement
{
public:
	PredicatedInstruction() {}
	PredicatedInstruction(Register<PredicateType> *predicate, bool negate = false) : m_predicate(predicate), m_negatePredicate(negate) {}

	// Properties

	bool HasPredicate() const { return m_predicate != nullptr; }

	std::pair<const Register<PredicateType> *, bool> GetPredicate() const { return {m_predicate, m_negatePredicate}; }
	std::pair<Register<PredicateType> *, bool> GetPredicate() { return {m_predicate, m_negatePredicate}; }

	void SetPredicate(Register<PredicateType> *predicate, bool negate = false)
	{
		m_predicate = predicate;
		m_negatePredicate = negate;
	}

	// Formatting

	std::string GetPrefix() const override
	{
		if (m_predicate != nullptr)
		{
			if (m_negatePredicate)
			{
				return "@!" + m_predicate->GetName();
			}
			return "@" + m_predicate->GetName();
		}
		return "";
	}

	json ToJSON() const override
	{
		json j = InstructionStatement::ToJSON();
		if (m_predicate != nullptr)
		{
			j["predicate"] = m_predicate->ToJSON();
			j["negate_predicate"] = m_negatePredicate;
		}
		return j;
	}

protected:
	Register<PredicateType> *m_predicate = nullptr;
	bool m_negatePredicate = false;
};

}
