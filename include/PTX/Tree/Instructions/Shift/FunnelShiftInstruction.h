#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

#include "PTX/Traversal/InstructionDispatch.h"

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

	FunnelShiftInstruction(const Register<T> *destination, const TypedOperand<T> *sourceA, const TypedOperand<T> *sourceB, const TypedOperand<UInt32Type> *shift, Direction direction, Mode mode) : InstructionBase_3<T, T, T, UInt32Type>(destination, sourceA, sourceB, shift), m_direction(direction), m_mode(mode) {}

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
		return code + T::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);

	Direction m_direction;
	Mode m_mode;
};

DispatchImplementation(FunnelShiftInstruction)

}
