#pragma once

#include "HorseIR/Tree/Node.h"

namespace HorseIR {

class Type : public Node
{
public:
	enum class Kind {
		Basic,
		Function,
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
