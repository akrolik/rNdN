#pragma once

#include <string>

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class BasicType : public Type
{
public:
	enum class BasicKind {
		Boolean,
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
		Datetime,
		Date,
		Month,
		Minute,
		Second,
		Time
	};

	constexpr static Type::Kind TypeKind = Type::Kind::Basic;

	BasicType(BasicKind kind) : Type(TypeKind), m_basicKind(kind) {}

	BasicType *Clone() const override
	{
		return new BasicType(m_basicKind);
	}

	static std::string BasicKindString(BasicKind kind)
	{
		switch (kind)
		{
			case BasicKind::Boolean:
				return "bool";
			case BasicKind::Char:
				return "char";
			case BasicKind::Int8:
				return "i8";
			case BasicKind::Int16:
				return "i16";
			case BasicKind::Int32:
				return "i32";
			case BasicKind::Int64:
				return "i64";
			case BasicKind::Float32:
				return "f32";
			case BasicKind::Float64:
				return "f64";
			case BasicKind::Complex:
				return "complex";
			case BasicKind::Symbol:
				return "sym";
			case BasicKind::String:
				return "string";
			case BasicKind::Datetime:
				return "dt";
			case BasicKind::Date:
				return "date";
			case BasicKind::Month:
				return "month";
			case BasicKind::Minute:
				return "minute";
			case BasicKind::Second:
				return "second";
			case BasicKind::Time:
				return "time";
			default:
				return "<unknown>";
		}
	}

	BasicKind GetBasicKind() const { return m_basicKind; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

	bool operator==(const BasicType& other) const
	{
		return (m_basicKind == other.m_basicKind);
	}

	bool operator!=(const BasicType& other) const
	{
		return (m_basicKind != other.m_basicKind);
	}

protected:
	BasicKind m_basicKind;
};

}
