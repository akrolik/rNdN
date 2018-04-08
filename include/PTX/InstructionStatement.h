#pragma once

#include "PTX/Statement.h"
#include "PTX/PredicateRegister.h"

namespace PTX {

class InstructionStatement : public Statement
{
public:

private:
	std::string m_label;
	PredicateRegister *m_predicate = nullptr;
	bool m_negatePredicate = false;
};

}
