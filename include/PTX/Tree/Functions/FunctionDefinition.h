#pragma once

#include "PTX/Tree/Functions/FunctionDeclaration.h"

#include "PTX/Tree/Statements/StatementList.h"
#include "PTX/Tree/Statements/Statement.h"

namespace PTX {

class VoidType;

template<class R>
class FunctionDefinition : public FunctionDeclaration<R>, public StatementList
{
public:
	// Formatting

	json ToJSON() const override
	{
		json j = FunctionDeclaration<R>::ToJSON();
		j["statements"] = StatementList::ToJSON();
		return j;
	}

	// Visitors

	void Accept(Visitor& visitor) override
	{
		if constexpr(std::is_same<R, VoidType>::value)
		{
			visitor.Visit(this);
		}
	}

	void Accept(ConstVisitor& visitor) const override
	{
		if constexpr(std::is_same<R, VoidType>::value)
		{
			visitor.Visit(this);
		}
	}

	void Accept(HierarchicalVisitor& visitor) override
	{
		if constexpr(std::is_same<R, VoidType>::value)
		{
			if (visitor.VisitIn(this))
			{
				for (auto& parameter : FunctionDeclaration<R>::m_parameters)
				{
					parameter->Accept(visitor);
				}
				for (auto& statement : m_statements)
				{
					statement->Accept(visitor);
				}
			}
			visitor.VisitOut(this);
		}
	}

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
