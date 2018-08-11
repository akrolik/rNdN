#pragma once

#include <string>

#include "HorseIR/Tree/Node.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class Declaration : public Node
{
public:
	Declaration(const std::string& name, Type *type) : m_name(name), m_type(type) {}

	const std::string& GetName() const { return m_name; }
	Type *GetType() const { return m_type; }

	std::string ToString() const override
	{
		return m_name + ":" + m_type->ToString();
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

private:
	std::string m_name;
	Type *m_type = nullptr;
};

}
