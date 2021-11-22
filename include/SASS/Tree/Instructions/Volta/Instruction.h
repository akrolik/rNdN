#pragma once

#include "SASS/Tree/Instructions/Instruction.h"

namespace SASS {
namespace Volta {

class Instruction : public SASS::Instruction
{
public:
	virtual std::uint64_t BinaryOpCode() const = 0;

	std::uint64_t ToBinary() const override
	{
		std::uint64_t code = 0x0;
		code |= BinaryOpCode();
		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		// Schedule embedded directly in instruction format

		std::uint64_t code = 0x0;
		code |= (static_cast<std::uint64_t>(m_schedule.ToBinary()) << 41);
		return code;
	}

};

}
}
