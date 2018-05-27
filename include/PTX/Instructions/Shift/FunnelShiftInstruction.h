#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

class FunnelShiftInstruction : public InstructionBase_3<Bit32Type, Bit32Type, Bit32Type, UInt32Type>
{
public:
	enum Direction {
		Left,
		Right
	};

	enum Mode {
		Clamp,
		Wrap
	};

	FunnelShiftInstruction(Register<Bit32Type> *destination, Operand<Bit32Type> *sourceA, Operand<Bit32Type> *sourceB, Operand<UInt32Type> *shift, Direction direction, Mode mode) : InstructionBase_3<Bit32Type, Bit32Type, Bit32Type, UInt32Type>(destination, sourceA, sourceB, shift), m_direction(direction), m_mode(mode) {}

	std::string OpCode() const
	{
		std::string code = "shf";
		if (m_direction == Direction::Left)
		{
			code += ".l";
		}
		else
		{
			code += ".r";
		}
		if (m_mode == Mode::Clamp)
		{
			code += ".clamp";
		}
		else
		{
			code += ".wrap";
		}
		return code + Bit32Type::Name();
	}

private:
	Direction m_direction;
	Mode m_mode;
};

}
