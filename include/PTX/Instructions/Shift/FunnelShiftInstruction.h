#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

namespace PTX {

class FunnelShiftInstruction : public PredicatedInstruction
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

	FunnelShiftInstruction(Register<Bit32Type> *destination, Operand<Bit32Type> *sourceA, Operand<Bit32Type> *sourceB, Operand<UInt32Type> *shift, Direction direction, Mode mode) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_shift(shift), m_direction(direction), m_mode(mode) {}

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
	
	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString() + ", " + m_shift->ToString();
	}

private:
	Register<Bit32Type> *m_destination = nullptr;
	Operand<Bit32Type> *m_sourceA = nullptr;
	Operand<Bit32Type> *m_sourceB = nullptr;
	Operand<UInt32Type> *m_shift = nullptr;
	Direction m_direction;
	Mode m_mode;
};

}
