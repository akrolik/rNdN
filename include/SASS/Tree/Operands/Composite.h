#pragma once

#include "SASS/Tree/Operands/Operand.h"

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
