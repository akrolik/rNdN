#pragma once

#include <string>

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class PrimitiveType : public Type
{
public:
	enum Kind {
		Wildcard,
		Bool,
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

	PrimitiveType(Kind kind) : Type(Type::Kind::Primitive), m_kind(kind) {}

	std::string ToString() const override
	{
		switch (m_kind)
		{
			case Wildcard:
				return "?";
			case Bool:
				return "bool";
			case Int8:
				return "i8";
			case Int16:
				return "i16";
			case Int32:
				return "i32";
			case Int64:
				return "i64";
			case Float32:
				return "f32";
			case Float64:
				return "f64";
			case Complex:
				return "complex";
			case Symbol:
				return "sym";
			case String:
				return "string";
			case Table:
				return "table";
			default:
				return "<unknown>";
		}
	}

	Kind GetKind() const { return m_kind; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }

private:
	const Kind m_kind;
};

}
