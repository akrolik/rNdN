#pragma once

#include "HorseIR/Tree/Node.h"

namespace HorseIR {

class Type : public Node
{
public:
	enum class Kind {
		Primitive,
		List
	};

	Type(Kind kind) : m_kind(kind) {}

	Kind GetKind() const { return m_kind; };

	bool operator==(const Type& other) const;
	bool operator!=(const Type& other) const;

private:
	const Kind m_kind;
};

}

#include "HorseIR/Tree/Types/ListType.h"
#include "HorseIR/Tree/Types/PrimitiveType.h"

namespace HorseIR {

inline bool Type::operator==(const Type& other) const
{
	bool sameKind = (m_kind == other.m_kind);
	if (sameKind)
	{
		switch (m_kind)
		{
			case Kind::Primitive:
				return static_cast<const PrimitiveType&>(*this) == static_cast<const PrimitiveType&>(other);
			case Kind::List:
				return static_cast<const ListType&>(*this) == static_cast<const ListType&>(other);
		}
	}
	return false;
}

inline bool Type::operator!=(const Type& other) const
{
	bool sameKind = (m_kind == other.m_kind);
	if (sameKind)
	{
		switch (m_kind)
		{
			case Kind::Primitive:
				return static_cast<const PrimitiveType&>(*this) != static_cast<const PrimitiveType&>(other);
			case Kind::List:
				return static_cast<const ListType&>(*this) != static_cast<const ListType&>(other);
		}
	}
	return true;
}

}
