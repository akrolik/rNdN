#pragma once

#include <string>

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class ListType : public Type
{
public:
	ListType(Type *elementType) : m_elementType(elementType) {}

	std::string ToString() const override
	{
		return "list<" + m_elementType->ToString() + ">";
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	Type *m_elementType = nullptr;
};

}
