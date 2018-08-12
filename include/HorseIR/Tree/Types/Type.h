#pragma once

#include "HorseIR/Tree/Node.h"

namespace HorseIR {

class Type : public Node
{
public:
	enum class Kind {
		Basic,
		List,
		Table,
		Dictionary,
		Enumeration,
		KeyedTable
	};

	Kind GetKind() const { return m_kind; };

	bool operator==(const Type& other) const;
	bool operator!=(const Type& other) const;

protected:
	Type(Kind kind) : m_kind(kind) {}

	const Kind m_kind;
};

}

#include "HorseIR/Tree/Types/BasicType.h"
#include "HorseIR/Tree/Types/DictionaryType.h"
#include "HorseIR/Tree/Types/ListType.h"
#include "HorseIR/Tree/Types/TableType.h"

namespace HorseIR {

inline bool Type::operator==(const Type& other) const
{
	if (m_kind == other.m_kind)
	{
		switch (m_kind)
		{
			case Kind::Basic:
				return static_cast<const BasicType&>(*this) == static_cast<const BasicType&>(other);
			case Kind::List:
				return static_cast<const ListType&>(*this) == static_cast<const ListType&>(other);
			case Kind::Table:
				return static_cast<const TableType&>(*this) == static_cast<const TableType&>(other);
			case Kind::Dictionary:
				return static_cast<const DictionaryType&>(*this) == static_cast<const DictionaryType&>(other);
		}
	}
	return false;
}

inline bool Type::operator!=(const Type& other) const
{
	if (m_kind == other.m_kind)
	{
		switch (m_kind)
		{
			case Kind::Basic:
				return static_cast<const BasicType&>(*this) != static_cast<const BasicType&>(other);
			case Kind::List:
				return static_cast<const ListType&>(*this) != static_cast<const ListType&>(other);
			case Kind::Table:
				return static_cast<const TableType&>(*this) != static_cast<const TableType&>(other);
			case Kind::Dictionary:
				return static_cast<const DictionaryType&>(*this) != static_cast<const DictionaryType&>(other);
		}
	}
	return true;
}

}
