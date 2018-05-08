#pragma once

#include <sstream>

#include "PTX/Statements/Statement.h"
#include "PTX/Operands/PredicateRegister.h"

namespace PTX {

class InstructionStatement : public Statement
{
public:
	void SetLabel(std::string label) { m_label = label; }
	void SetPredicate(PredicateRegister *predicate, bool negate) { m_predicate = predicate; m_negatePredicate = negate; }

	std::string ToString()
	{
		std::ostringstream code;
		if (m_label.length() > 0)
		{
			code << m_label << ": ";
		}
		if (m_predicate != nullptr)
		{
			code << "@";
			if (m_negatePredicate)
			{
				code << "!";
			}
			code << m_predicate->VariableName() << " ";
		}
		code << "\t" << OpCode() << " " << Operands() << ";" << std::endl;
		return code.str();
	}

	virtual std::string OpCode() = 0;
	virtual std::string Operands() = 0;

private:
	std::string m_label;
	PredicateRegister *m_predicate = nullptr;
	bool m_negatePredicate = false;
};

}
