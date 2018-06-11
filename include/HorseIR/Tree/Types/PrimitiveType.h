#pragma once

#include <string>

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class PrimitiveType : public Type
{
public:
	enum Type {
		Int64,
		String
	};

	PrimitiveType(Type type) : m_type(type) {}

	std::string ToString() const override
	{
		switch (m_type)
		{
			case Int64:
				return "i64";
			case String:
				return "string";
			default:
				return "<unknown>";
		}
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	Type m_type;
};

}
