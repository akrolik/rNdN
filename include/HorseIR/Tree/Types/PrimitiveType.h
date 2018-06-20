#pragma once

#include <string>

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class PrimitiveType : public Type
{
public:
	enum Type {
		Wildcard,
		Bool,
		Char,
		Int8,
		Int16,
		Int32,
		Int64,
		Float32,
		Float64,
		Complex,
		Symbol,
		String,
		Table
	};

	PrimitiveType(Type type) : m_type(type) {}

	std::string ToString() const override
	{
		switch (m_type)
		{
			case Int8:
				return "i8";
			case Int64:
				return "i64";
			case String:
				return "string";
			default:
				return "<unknown>";
		}
	}

	Type GetType() const { return m_type; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	Type m_type;
};

}
