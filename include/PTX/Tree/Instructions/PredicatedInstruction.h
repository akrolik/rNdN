#pragma once

#include "PTX/Tree/Statements/InstructionStatement.h"

#include <utility>

#include "PTX/Tree/Type.h"
#include "PTX/Tree/Operands/Variables/Registers/Register.h"

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

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor& visitor) override
	{
		if (visitor.VisitIn(this))
		{
			if (m_predicate != nullptr)
			{
				static_cast<Operand *>(m_predicate)->Accept(visitor);
			}
			for (auto& operand : GetOperands())
			{
				operand->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			if (m_predicate != nullptr)
			{
				static_cast<const Operand *>(m_predicate)->Accept(visitor);
			}
			for (const auto& operand : GetOperands())
			{
				operand->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

protected:
	Register<PredicateType> *m_predicate = nullptr;
	bool m_negatePredicate = false;
};

}
