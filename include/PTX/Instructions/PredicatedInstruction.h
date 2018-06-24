#pragma once

#include "PTX/Statements/InstructionStatement.h"

#include "PTX/Type.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

class PredicatedInstruction : public InstructionStatement
{
public:
	void SetPredicate(const Register<PredicateType> *predicate, bool negate = false) { m_predicate = predicate; m_negatePredicate = negate; }

	std::string ToString() const override
	{
		std::ostringstream code;
		if (m_predicate != nullptr)
		{
			code << "@";
			if (m_negatePredicate)
			{
				code << "!";
			}
			code << m_predicate->GetName() << " ";
		}
		code << InstructionStatement::ToString();
		return code.str();
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

private:
	const Register<PredicateType> *m_predicate = nullptr;
	bool m_negatePredicate = false;
};

}
