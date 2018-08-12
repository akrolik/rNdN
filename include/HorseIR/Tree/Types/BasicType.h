#pragma once

#include <string>

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class BasicType : public Type
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
		Datetime,
		Date,
		Month,
		Time,
		Minute,
		Second,
		Function
	};

	BasicType(Kind kind) : Type(Type::Kind::Basic), m_kind(kind) {}

	static std::string KindString(Kind kind)
	{
		switch (kind)
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
				return "clex";
			case Kind::Symbol:
				return "sym";
			case Kind::String:
				return "string";
			case Kind::Month:
				return "mm";
			case Kind::Date:
				return "dd";
			case Kind::Datetime:
				return "dt";
			case Kind::Minute:
				return "mn";
			case Kind::Second:
				return "ss";
			case Kind::Time:
				return "tt";
			case Kind::Function:
				return "func";
			default:
				return "<unknown>";
		}
	}

	static std::string KindName(Kind kind)
	{
		switch (kind)
		{
			case Kind::Wildcard:
				return "wildcard";
			case Kind::Bool:
				return "bool";
			case Kind::Int8:
				return "char";
			case Kind::Int16:
				return "short";
			case Kind::Int32:
				return "int";
			case Kind::Int64:
				return "long";
			case Kind::Float32:
				return "float";
			case Kind::Float64:
				return "double";
			case Kind::Complex:
				return "complex";
			case Kind::Symbol:
				return "symbol";
			case Kind::String:
				return "string";
			case Kind::Month:
				return "month";
			case Kind::Date:
				return "datee";
			case Kind::Datetime:
				return "datetime";
			case Kind::Minute:
				return "minute";
			case Kind::Second:
				return "second";
			case Kind::Time:
				return "time";
			case Kind::Function:
				return "function";
			default:
				return "<unknown>";
		}
	}

	std::string ToString() const override
	{
		return KindString(m_kind);
	}

	Kind GetKind() const { return m_kind; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	bool operator==(const BasicType& other) const
	{
		return m_kind == other.m_kind;
	}

	bool operator!=(const BasicType& other) const
	{
		return m_kind != other.m_kind;
	}

private:
	const Kind m_kind;
};

}
