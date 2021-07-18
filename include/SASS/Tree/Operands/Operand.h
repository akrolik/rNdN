#pragma once

#include "SASS/Tree/Node.h"

namespace SASS {

class Operand : public Node
{
public:
	// Binary/Formatting

	virtual std::string ToString() const = 0;
	virtual std::uint64_t ToBinary() const = 0;

	enum class Kind {
		Register,
		SpecialRegister,
		Predicate,
		Flag,
		Immediate,
		Constant,
		Address
	};

	Kind GetKind() const { return m_kind; }

	virtual std::uint64_t ToBinary(std::uint8_t truncate) const
	{
		return ToBinary();
	}

protected:
	Operand(Kind kind) : m_kind(kind) {}
	Kind m_kind;
};

}
