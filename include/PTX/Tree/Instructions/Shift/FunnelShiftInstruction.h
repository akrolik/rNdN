#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

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

	FunnelShiftInstruction(const Register<Bit32Type> *destination, const TypedOperand<Bit32Type> *sourceA, const TypedOperand<Bit32Type> *sourceB, const TypedOperand<UInt32Type> *shift, Direction direction, Mode mode) : InstructionBase_3<Bit32Type, Bit32Type, Bit32Type, UInt32Type>(destination, sourceA, sourceB, shift), m_direction(direction), m_mode(mode) {}

	Direction GetDirection() const { return m_direction; }
	void SetDirection(Direction direction) { m_direction = direction; }

	Mode GetMode() const { return m_mode; }
	void SetMode(Mode mode) { m_mode = mode; }

	static std::string Mnemonic() { return "shf"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic();
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
