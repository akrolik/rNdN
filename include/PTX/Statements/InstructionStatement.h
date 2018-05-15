#pragma once

#include <sstream>

#include "PTX/Statements/Statement.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

class InstructionStatement : public Statement
{
public:
	void SetPredicate(Register<PredicateType> *predicate, bool negate = false) { m_predicate = predicate; m_negatePredicate = negate; }

	std::string ToString() const
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
		code << "\t" << OpCode();
		std::string operands = Operands();
		if (operands.length() > 0)
		{
			code << " " << operands;
		}
		return code.str();
	}

	std::string Terminator() const { return ";"; }

	virtual std::string OpCode() const = 0;
	virtual std::string Operands() const = 0;

private:
	Register<PredicateType> *m_predicate = nullptr;
	bool m_negatePredicate = false;
};

}
