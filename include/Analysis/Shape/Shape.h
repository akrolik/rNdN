#pragma once

#include <sstream>

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace Analysis {

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

		friend std::ostream& operator<<(std::ostream& os, const Size& shape);

		virtual void Print(std::ostream& os) const = 0;

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

		void Print(std::ostream& os) const override
		{
			os << m_value;
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

		void Print(std::ostream& os) const override
		{
			os << "`" << m_symbol;
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
		CompressedSize(const HorseIR::Operand *predicate, const Size *size) : Size(Size::Kind::Compressed), m_predicate(predicate), m_size(size) {}

		void Print(std::ostream& os) const override
		{
			os << *m_size << "[" << HorseIR::PrettyPrinter::PrettyString(m_predicate) << "]";
		}

		bool operator==(const CompressedSize& other) const
		{
			return (*m_predicate == *other.m_predicate && *m_size == *other.m_size);
		}

		bool operator!=(const CompressedSize& other) const
		{
			return !(*this == other);
		}

		const HorseIR::Operand *m_predicate = nullptr;
		const Size *m_size = nullptr;;
	};

	struct DynamicSize : Size
	{
		DynamicSize(const Size *size1, const Size *size2) : Size(Size::Kind::Dynamic), m_size1(size1), m_size2(size2) {}
		DynamicSize(const HorseIR::Expression *expression, unsigned int tag = 0) : Size(Size::Kind::Dynamic), m_expression(expression), m_tag(tag) {}

		void Print(std::ostream& os) const override
		{
			if (m_size1 != nullptr)
			{
				os << "Dynamic/Merge<" << *m_size1 << ", " << *m_size2 << ">";
			}
			else if (m_tag == 0)
			{
				os << "Dynamic/" << HorseIR::PrettyPrinter::PrettyString(m_expression);
			}
			else
			{
				os << "Dynamic/" << HorseIR::PrettyPrinter::PrettyString(m_expression) << "/" << m_tag;
			}
		}

		bool operator==(const DynamicSize& other) const
		{
			if (m_size1 != nullptr && other.m_size1 != nullptr && m_size2 != nullptr && other.m_size2 != nullptr)
			{
				return (*m_size1 == *other.m_size1 && *m_size2 == *other.m_size2);
			}
			return (m_expression == other.m_expression && m_tag == other.m_tag);
		}

		bool operator!=(const DynamicSize& other) const
		{
			return !(*this == other);
		}

		const Size *m_size1 = nullptr;
		const Size *m_size2 = nullptr;

		const HorseIR::Expression *m_expression = nullptr;
		unsigned int m_tag = -1;
	};

	enum class Kind {
		Wildcard,
		Vector,
		List,
		Table,
		KeyedTable,
		Dictionary,
		Enumeration
	};

	Kind GetKind() const { return m_kind; }

	friend std::ostream& operator<<(std::ostream& os, const Shape& shape);

	virtual void Print(std::ostream& os) const = 0;

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

	WildcardShape(const Shape *shape1, const Shape *shape2) : Shape(Shape::Kind::Wildcard), m_shape1(shape1), m_shape2(shape2) {}
	WildcardShape(const HorseIR::Expression *expression) : Shape(Shape::Kind::Wildcard), m_expression(expression) {}

	void Print(std::ostream& os) const override
	{
		if (m_shape1 != nullptr)
		{
			os << "*<" << *m_shape1 << ", " << *m_shape2 << ">";
		}
		else
		{
			os << "*";
		}
	}

	bool operator==(const WildcardShape& other) const
	{
		if (m_shape1 != nullptr && other.m_shape1 != nullptr && m_shape2 != nullptr && other.m_shape2 != nullptr)
		{
			return (*m_shape1 == *other.m_shape1 && *m_shape2 == *other.m_shape2);
		}
		return (m_expression == other.m_expression);
	}

	bool operator!=(const WildcardShape& other) const
	{
		return !(*this == other);
	}

private:
	const Shape *m_shape1 = nullptr;
	const Shape *m_shape2 = nullptr;

	const HorseIR::Expression *m_expression = nullptr;
};

class VectorShape : public Shape
{
public:
	constexpr static Kind ShapeKind = Kind::Vector;

	VectorShape(const Size *size) : Shape(Shape::Kind::Vector), m_size(size) {}

	const Size *GetSize() const { return m_size; }

	void Print(std::ostream& os) const override
	{
		os << "VectorShape<" << *m_size << ">";
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

	void Print(std::ostream& os) const override
	{
		os << "ListShape<" << *m_listSize << ", " << *m_elementShape << ">";
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

	void Print(std::ostream& os) const override
	{
		os << "TableShape<" << *m_columnsSize << ", " << *m_rowsSize << ">";
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

class KeyedTableShape : public Shape
{
public:
	constexpr static Kind ShapeKind = Kind::KeyedTable;

	KeyedTableShape(const TableShape *keyShape, const TableShape *valueShape) : Shape(Shape::Kind::KeyedTable), m_keyShape(keyShape), m_valueShape(valueShape) {}

	const TableShape *GetKeyShape() const { return m_keyShape; }
	const TableShape *GetValueShape() const { return m_valueShape; }

	void Print(std::ostream& os) const override
	{
		os << "KTableShape<" << *m_keyShape << ", " << *m_valueShape << ">";
	}

	bool operator==(const KeyedTableShape& other) const
	{
		return (*m_keyShape == *other.m_keyShape && *m_valueShape == *other.m_valueShape);
	}

	bool operator!=(const KeyedTableShape& other) const
	{
		return !(*this == other);
	}

private:
	const TableShape *m_keyShape = nullptr;
	const TableShape *m_valueShape = nullptr;
};

class DictionaryShape : public Shape
{
public:
	constexpr static Kind ShapeKind = Kind::Dictionary;

	DictionaryShape(const Shape *keyShape, const Shape *valueShape) : Shape(Shape::Kind::Dictionary), m_keyShape(keyShape), m_valueShape(valueShape) {}

	const Shape *GetKeyShape() const { return m_keyShape; }
	const Shape *GetValueShape() const { return m_valueShape; }

	void Print(std::ostream& os) const override
	{
		os << "DictionaryShape<" << *m_keyShape << ", " << *m_valueShape << ">";
	}

	bool operator==(const DictionaryShape& other) const
	{
		return (*m_keyShape == *other.m_keyShape && *m_valueShape == *other.m_valueShape);
	}

	bool operator!=(const DictionaryShape& other) const
	{
		return !(*this == other);
	}

private:
	const Shape *m_keyShape = nullptr;
	const Shape *m_valueShape = nullptr;
};

class EnumerationShape : public Shape
{
public:
	constexpr static Kind ShapeKind = Kind::Enumeration;

	EnumerationShape(const Size *mapSize) : Shape(Shape::Kind::Enumeration), m_mapSize(mapSize) {}

	const Size *GetMapSize() const { return m_mapSize; }

	void Print(std::ostream& os) const override
	{
		os << "EnumerateShape<" << *m_mapSize << ">";
	}

	bool operator==(const EnumerationShape& other) const
	{
		return (*m_mapSize == *other.m_mapSize);
	}

	bool operator!=(const EnumerationShape& other) const
	{
		return !(*this == other);
	}

private:
	const Size *m_mapSize = nullptr;
};

inline std::ostream& operator<<(std::ostream& os, const Shape& shape)
{
	shape.Print(os);
	return os;
}

inline std::ostream& operator<<(std::ostream& os, const Shape::Size& size)
{
	size.Print(os);
	return os;
}

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
			case Kind::KeyedTable:
				return static_cast<const KeyedTableShape&>(*this) == static_cast<const KeyedTableShape&>(other);
			case Kind::Dictionary:
				return static_cast<const DictionaryShape&>(*this) == static_cast<const DictionaryShape&>(other);
			case Kind::Enumeration:
				return static_cast<const EnumerationShape&>(*this) == static_cast<const EnumerationShape&>(other);
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
