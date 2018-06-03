#pragma once

#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class ListType : public Type
{
public:
	ListType(Type *elementType) : m_elementType(elementType) {}

	std::string ToString() const
	{
		return "list<" + m_elementType->ToString() + ">";
	}

private:
	Type *m_elementType = nullptr;
};

}
