#pragma once

#include "HorseIR/Tree/Node.h"

namespace Analysis { class ShapeUtils; }

namespace HorseIR {

class Type : public Node
{
public:
	virtual Type *Clone() const override = 0;

	bool operator==(const Type& other) const;
	bool operator!=(const Type& other) const;

	friend class TypeUtils;
	friend class Analysis::ShapeUtils;

protected:
	enum class Kind {
		Wildcard,
		Basic,
		Function,
		List,
		Table,
		Dictionary,
		Enumeration,
		KeyedTable
	};

	Type(Kind kind) : m_kind(kind) {}
	Kind m_kind;
};

}
