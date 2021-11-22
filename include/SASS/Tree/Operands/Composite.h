#pragma once

#include "SASS/Tree/Operands/Operand.h"

namespace SASS {

class Composite : public Operand
{
public:
	using Operand::Operand;

	virtual bool GetOpModifierNegate() const { return false; }

	virtual std::uint64_t ToAbsoluteBinary(std::uint8_t truncate) const
	{
		return ToBinary();
	}

	virtual std::uint64_t ToTruncatedBinary(std::uint8_t truncate) const
	{
		return ToBinary();
	}
};

}
