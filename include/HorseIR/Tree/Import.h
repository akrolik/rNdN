#pragma once

#include <string>

#include "HorseIR/Tree/ModuleContent.h"

#include "HorseIR/Tree/Expressions/ModuleIdentifier.h"
#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class Import : public ModuleContent
{
public:
	Import(ModuleIdentifier *identifier) : m_identifier(identifier) {}

	ModuleIdentifier *GetIdentifier() const { return m_identifier; }
	void SetIdentifier(ModuleIdentifier *identifier) { m_identifier = identifier; }

	std::string ToString() const override
	{
		return "import " + m_identifier->ToString() + ";";
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

private:
	ModuleIdentifier *m_identifier = nullptr;
};

}
