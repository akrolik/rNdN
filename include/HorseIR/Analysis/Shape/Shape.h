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
		Wildcard,
		Vector,
		List,
		Table
	};

	Kind GetKind() const { return m_kind; }

	virtual std::string ToString() const = 0;

	bool operator==(const Shape& other) const;
	bool operator!=(const Shape& other) const
	{
		return !(*this == other);
	}

protected:
	Shape(Kind kind) : m_kind(kind) {}

	const Kind m_kind;
};

class WildcardShape : public Shape
{
public:
	constexpr static Kind ShapeKind = Kind::Wildcard;

	WildcardShape() : Shape(Shape::Kind::Wildcard) {}

	std::string ToString() const override
	{
		return "*";
	}

	bool operator==(const WildcardShape& other) const
	{
		return false;
	}

	bool operator!=(const WildcardShape& other) const
	{
		return true;
	}

private:
	const Size *m_size = nullptr;
};
class VectorShape : public Shape
{
public:
	constexpr static Kind ShapeKind = Kind::Vector;

	VectorShape(const Size *size) : Shape(Shape::Kind::Vector), m_size(size) {}

	const Size *GetSize() const { return m_size; }

	std::string ToString() const override
	{
		return "VectorShape<" + m_size->ToString() + ">";
	}

	bool operator==(const VectorShape& other) const
	{
		return (*m_size == *other.m_size);
	}

	bool operator!=(const VectorShape& other) const
	{
		return !(*this == other);
	}

private:
	const Size *m_size = nullptr;
};

class ListShape : public Shape
{
public:
	constexpr static Kind ShapeKind = Kind::List;

	ListShape(const Size *listSize, const Shape *elementShape) : Shape(Shape::Kind::List), m_listSize(listSize), m_elementShape(elementShape) {}

	const Size *GetListSize() const { return m_listSize; }
	const Shape *GetElementShape() const { return m_elementShape; }

	std::string ToString() const override
	{
		return "ListShape<" + m_listSize->ToString() + ", " + m_elementShape->ToString() + ">";
	}

	bool operator==(const ListShape& other) const
	{
		return (*m_listSize == *other.m_listSize && *m_elementShape == *other.m_elementShape);
	}

	bool operator!=(const ListShape& other) const
	{
		return !(*this == other);
	}

private:
	const Size *m_listSize = nullptr;
	const Shape *m_elementShape = nullptr;
};

class TableShape : public Shape
{
public:
	constexpr static Kind ShapeKind = Kind::Table;

	TableShape(const Size *columnsSize, const Size *rowsSize) : Shape(Shape::Kind::Table), m_columnsSize(columnsSize), m_rowsSize(rowsSize) {}

	const Size *GetColumnsSize() const { return m_columnsSize; }
	const Size *GetRowsSize() const { return m_rowsSize; }

	std::string ToString() const override
	{
		return "TableShape<" + m_columnsSize->ToString() + ", " + m_rowsSize->ToString() + ">";
	}

	bool operator==(const TableShape& other) const
	{
		return (*m_columnsSize == *other.m_columnsSize && *m_rowsSize == *other.m_rowsSize);
	}

	bool operator!=(const TableShape& other) const
	{
		return !(*this == other);
	}

private:
	const Size *m_columnsSize = nullptr;
	const Size *m_rowsSize = nullptr;
};

inline bool Shape::operator==(const Shape& other) const
{
	if (m_kind == other.m_kind)
	{
		switch (m_kind)
		{
			case Kind::Wildcard:
				return static_cast<const WildcardShape&>(*this) == static_cast<const WildcardShape&>(other);
			case Kind::Vector:
				return static_cast<const VectorShape&>(*this) == static_cast<const VectorShape&>(other);
			case Kind::List:
				return static_cast<const ListShape&>(*this) == static_cast<const ListShape&>(other);
			case Kind::Table:
				return static_cast<const TableShape&>(*this) == static_cast<const TableShape&>(other);
		}
	}
	return false;
}
                 
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
