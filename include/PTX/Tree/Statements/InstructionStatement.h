#pragma once

#include "PTX/Tree/Statements/Statement.h"
#include "PTX/Tree/Operands/Operand.h"

#include "PTX/Traversal/ConstInstructionVisitor.h"

namespace PTX {

class InstructionStatement : public Statement
{
public:
	// Properties

	virtual std::string GetOpCode() const = 0;
	virtual std::vector<const Operand *> GetOperands() const = 0;
	virtual std::vector<Operand *> GetOperands() = 0;

	// Formatting

	virtual std::string GetPrefix() const { return ""; }

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::InstructionStatement";
		j["opcode"] = GetOpCode();
		for (const auto& operand : GetOperands())
		{
			j["operands"].push_back(operand->ToJSON());
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
			for (const auto& operand : GetOperands())
			{
				operand->Accept(visitor);
			}
		}
		visitor.VisitOut(this);
	}

	virtual void Accept(ConstInstructionVisitor &visitor) const = 0;
};

}
