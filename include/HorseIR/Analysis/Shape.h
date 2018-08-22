#pragma once

#include <string>

#include "HorseIR/Tree/Expressions/Expression.h"

namespace HorseIR {

class Shape
{
public:
	struct Size
	{
		enum class Kind {
			Constant,
			Symbol,
			Compressed,
			Dynamic
		};

		Size(Kind kind) : m_kind(kind) {}

		virtual std::string ToString() const = 0;

		bool operator==(const Size& other) const;
		bool operator!=(const Size& other) const
		{
			return !(*this == other);
		}

		const Kind m_kind;
	};

	struct ConstantSize : Size
	{
		ConstantSize(long value) : Size(Size::Kind::Constant), m_value(value) {}

		std::string ToString() const override
		{
			return std::to_string(m_value);
		}

		bool operator==(const ConstantSize& other) const
		{
			return m_value == other.m_value;
		}

		bool operator!=(const ConstantSize& other) const
		{
			return !(*this == other);
		}

		const unsigned int m_value = 0;
	};

	struct SymbolSize : Size
	{
		SymbolSize(const std::string& symbol) : Size(Size::Kind::Symbol), m_symbol(symbol) {}

		std::string ToString() const override
		{
			return "Symbol/" + m_symbol;
		}

		bool operator==(const SymbolSize& other) const
		{
			return m_symbol == other.m_symbol;
		}

		bool operator!=(const SymbolSize& other) const
		{
			return !(*this == other);
		}

		const std::string m_symbol;
	};

	struct CompressedSize : Size
	{
		CompressedSize(const CallExpression *invocation, const Expression *predicate, const Size *size) : Size(Size::Kind::Compressed), m_invocation(invocation), m_predicate(predicate), m_size(size) {}

		std::string ToString() const override
		{
			return "Compressed[" + m_predicate->ToString() + "]/" + m_size->ToString();
		}

		bool operator==(const CompressedSize& other) const
		{
			return (m_invocation == other.m_invocation && m_predicate == other.m_predicate && m_size == m_size);
		}

		bool operator!=(const CompressedSize& other) const
		{
			return !(*this == other);
		}

		const CallExpression *m_invocation = nullptr;
		const Expression *m_predicate = nullptr;
		const Size *m_size = nullptr;;
	};

	struct DynamicSize : Size
	{
		DynamicSize(const CallExpression *invocation, const Expression *expression) : Size(Size::Kind::Dynamic), m_invocation(invocation), m_expression(expression) {}

		std::string ToString() const override
		{
			return "Dynamic/" + m_expression->ToString();
		}

		bool operator==(const DynamicSize& other) const
		{
			return (m_invocation == other.m_invocation && m_expression == other.m_expression);
		}

		bool operator!=(const DynamicSize& other) const
		{
			return !(*this == other);
		}

		const CallExpression *m_invocation = nullptr;
		const Expression *m_expression = nullptr;
	};

	enum class Kind {
		Vector,
		List,
		Table
	};

	Shape(Kind kind, const Size *size) : m_kind(kind), m_size(size) {}

	std::string ToString() const
	{
		std::string output = "Shape<";
		switch (m_kind)
		{
			case Kind::Vector:
				output += "vector";
				break;
			case Kind::List:
				output += "list";
				break;
			case Kind::Table:
				output += "table";
				break;
		}
		return output + ", " + m_size->ToString() + ">";
	}

	const Size *GetSize() const { return m_size; }

private:
	const Kind m_kind;
	const Size *m_size = nullptr;
};

inline bool Shape::Size::operator==(const Shape::Size& other) const
{
	if (m_kind == other.m_kind)
	{
		switch (m_kind)
		{
			case Kind::Constant:
				return static_cast<const Shape::ConstantSize&>(*this) == static_cast<const Shape::ConstantSize&>(other);
			case Kind::Symbol:
				return static_cast<const Shape::SymbolSize&>(*this) == static_cast<const Shape::SymbolSize&>(other);
			case Kind::Compressed:
				return static_cast<const Shape::CompressedSize&>(*this) == static_cast<const Shape::CompressedSize&>(other);
			case Kind::Dynamic:
				return static_cast<const Shape::DynamicSize&>(*this) == static_cast<const Shape::DynamicSize&>(other);
		}
	}
	return false;
}
                 
}
