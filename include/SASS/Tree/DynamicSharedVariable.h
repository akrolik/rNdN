#pragma once

#include "SASS/Tree/Node.h"

#include <string>

namespace SASS {

class DynamicSharedVariable : public Node
{
public:
	DynamicSharedVariable(const std::string& name) : m_name(name) {}
	
	// Properties

	const std::string& GetName() const { return m_name; }
	void SetName(const std::string& name) { m_name = name; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }
	
private:
	std::string m_name;
};

}
