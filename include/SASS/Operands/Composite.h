#pragma once

#include "SASS/Operands/Operand.h"

namespace SASS {

class Composite : public Operand
{
public:
	enum OpCodeKind {
		Base,
		Immediate,
		Constant
	};

	virtual OpCodeKind GetOpCodeKind() const { return OpCodeKind::Base; }
};

}
