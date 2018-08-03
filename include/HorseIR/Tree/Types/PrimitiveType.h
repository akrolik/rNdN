#pragma once

#include <string>

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class PrimitiveType : public Type
{
public:
	enum class Kind {
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
			case Kind::Wildcard:
				return "?";
			case Kind::Bool:
				return "bool";
			case Kind::Int8:
				return "i8";
			case Kind::Int16:
				return "i16";
			case Kind::Int32:
				return "i32";
			case Kind::Int64:
				return "i64";
			case Kind::Float32:
				return "f32";
			case Kind::Float64:
				return "f64";
			case Kind::Complex:
				return "complex";
			case Kind::Symbol:
				return "sym";
			case Kind::String:
				return "string";
			case Kind::Table:
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
