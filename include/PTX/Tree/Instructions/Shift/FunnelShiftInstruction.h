#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

DispatchInterface(FunnelShiftInstruction)

template<class T>
class FunnelShiftInstruction : DispatchInherit(FunnelShiftInstruction), public InstructionBase_3<T, T, T, UInt32Type>
{
public:
	REQUIRE_TYPE_PARAM(FunnelShiftInstruction,
		REQUIRE_EXACT(T, Bit32Type)
	);

	enum Direction {
		Left,
		Right
	};

	enum Mode {
		Clamp,
		Wrap
	};

	FunnelShiftInstruction(Register<T> *destination, TypedOperand<T> *sourceA, TypedOperand<T> *sourceB, TypedOperand<UInt32Type> *shift, Direction direction, Mode mode)
		: InstructionBase_3<T, T, T, UInt32Type>(destination, sourceA, sourceB, shift), m_direction(direction), m_mode(mode) {}

	// Properties

	Direction GetDirection() const { return m_direction; }
	void SetDirection(Direction direction) { m_direction = direction; }

	Mode GetMode() const { return m_mode; }
	void SetMode(Mode mode) { m_mode = mode; }

	// Formatting

	static std::string Mnemonic() { return "shf"; }

	std::string GetOpCode() const override
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
		return code + T::Name();
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);

	Direction m_direction;
	Mode m_mode;
};

DispatchImplementation(FunnelShiftInstruction)

}
