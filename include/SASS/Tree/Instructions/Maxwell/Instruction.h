#pragma once

#include "SASS/Tree/Instructions/Instruction.h"

namespace SASS {
namespace Maxwell {

class Instruction : public SASS::Instruction
{
public:
	// Binary

	virtual std::uint64_t BinaryOpCode() const = 0;
	virtual std::uint64_t BinaryOpModifiers() const { return 0; }
	virtual std::uint64_t BinaryOperands() const { return 0; }

	std::uint64_t ToBinary() const override
	{
		std::uint64_t code = 0x0;
		code |= BinaryOpCode();
		code |= BinaryOpModifiers();
		code |= BinaryOperands();
		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		return 0x0;
	}
};

}
}
