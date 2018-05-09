#pragma once

#include <sstream>

#include "PTX/Statements/Statement.h"
#include "PTX/Operands/Register/PredicateRegister.h"

namespace PTX {

class InstructionStatement : public Statement
{
public:
	void SetLabel(std::string label) { m_label = label; }
	void SetPredicate(PredicateRegister *predicate, bool negate) { m_predicate = predicate; m_negatePredicate = negate; }

	std::string ToString() const
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
			code << m_predicate->GetName() << " ";
		}
		code << "\t" << OpCode() << " " << Operands() << ";" << std::endl;
		return code.str();
	}

	virtual std::string OpCode() const = 0;
	virtual std::string Operands() const = 0;

private:
	std::string m_label;
	PredicateRegister *m_predicate = nullptr;
	bool m_negatePredicate = false;
};

}
