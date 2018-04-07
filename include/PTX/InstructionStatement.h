#pragma once

#include "PTX/Statement.h"

namespace PTX {

class InstructionStatement : Statement
{
public:
	virtual std::string ToString();

private:
	std::string m_label;
	PredicateRegister *m_predicate = nullptr;
	bool m_negatePredicate = false;

	std::vector<Operand> m_operands;
};

}
