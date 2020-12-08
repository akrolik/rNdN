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
	PredicatedInstruction(const Register<PredicateType> *predicate, bool negate = false) : m_predicate(predicate), m_negatePredicate(negate) {}

	bool HasPredicate() const { return m_predicate != nullptr; }
	std::pair<const Register<PredicateType> *, bool> GetPredicate() const { return {m_predicate, m_negatePredicate}; }
	void SetPredicate(const Register<PredicateType> *predicate, bool negate = false) { m_predicate = predicate; m_negatePredicate = negate; }

	std::string ToString(unsigned int indentation) const override
	{
		std::string code = std::string(indentation, '\t');
		if (m_predicate != nullptr)
		{
			code += "@";
			if (m_negatePredicate)
			{
				code += "!";
			}
			code += m_predicate->GetName() + " ";
		}
		return code + InstructionStatement::ToString(0);
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
	const Register<PredicateType> *m_predicate = nullptr;
	bool m_negatePredicate = false;
};

}
