#pragma once

#include "SASS/Operands/Operand.h"

namespace SASS {

class SpecialRegister : public Operand
{
public:
	enum class Kind {
		SR_TID_X,
		SR_CTAID_X
	};

	static std::string KindString(Kind kind)
	{
		switch (kind)
		{
			case Kind::SR_TID_X:
				return "SR_TID.X";
			case Kind::SR_CTAID_X:
				return "SR_CTAID.X";
		}
		return "<unknown>";
	}

	SpecialRegister(Kind kind) : m_kind(kind) {}

	std::string ToString() const override
	{
		return KindString(m_kind);
	}

private:
	Kind m_kind;
};

}
