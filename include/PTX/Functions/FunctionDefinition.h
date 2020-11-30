#pragma once

#include "PTX/Functions/FunctionDeclaration.h"

#include "PTX/Statements/StatementList.h"
#include "PTX/Statements/Statement.h"

namespace PTX {

class VoidType;

template<class R>
class FunctionDefinition : public FunctionDeclaration<R>, public StatementList
{
public:
	json ToJSON() const override
	{
		json j = FunctionDeclaration<R>::ToJSON();
		j["statements"] = StatementList::ToJSON();
		return j;
	}

	std::string ToString(unsigned int indentation) const override
	{
		return Function::ToString(indentation) + "\n" + StatementList::ToString(0);
	}

	// Visitors

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		if constexpr(std::is_same<R, VoidType>::value)
		{
			if (visitor.VisitIn(this))
			{
				for (const auto& parameter : FunctionDeclaration<R>::m_parameters)
				{
					parameter->Accept(visitor);
				}
				for (const auto& statement : m_statements)
				{
					statement->Accept(visitor);
				}
			}
			visitor.VisitOut(this);
		}
	}
};

}
