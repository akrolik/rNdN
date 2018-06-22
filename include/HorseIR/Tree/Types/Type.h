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

private:
	const Kind m_kind;
};

}
