#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

#include "PTX/Tree/Operands/Operand.h"
#include "PTX/Tree/Operands/Variables/Registers/Register.h"

namespace PTX {

DispatchInterface(PermuteInstruction)

template<class T, bool Assert = true>
class PermuteInstruction : DispatchInherit(PermuteInstruction), public InstructionBase_3<T>
{
public:
	REQUIRE_TYPE_PARAM(PermuteInstruction,
		REQUIRE_EXACT(T, Bit32Type)
	);

	enum class Mode {
		Generic,
		Forward4Extract,
		Backward4Extract,
		EdgeLeftClamp,
		EdgeRightClamp,
		Replicate8,
		Replicate16
	};

	static std::string ModeString(Mode mode)
	{
		switch (mode)
		{
			case Mode::Generic:
				return "";
			case Mode::Forward4Extract:
				return ".f4e";
			case Mode::Backward4Extract:
				return ".b4e";
			case Mode::EdgeLeftClamp:
				return ".ecl";
			case Mode::EdgeRightClamp:
				return ".ecr";
			case Mode::Replicate8:
				return ".rc8";
			case Mode::Replicate16:
				return ".rc16";
		}
		return ".<unknown>";
	}

	PermuteInstruction(Register<T> *destinationD, TypedOperand<T> *sourceA, TypedOperand<T> *sourceB, TypedOperand<T> *sourceC, Mode mode = Mode::Generic)
		: InstructionBase_3<T>(destinationD, sourceA, sourceB, sourceC), m_mode(mode) {}

	// Analysis properties

	bool HasSideEffect() const override { return false; }

	// Properties

	Mode GetMode() const { return m_mode; }
	void SetMode(Mode mode) { m_mode = mode; }

	// Formatting

	static std::string Mnemonic() { return "prmt"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + T::Name() + ModeString(m_mode);
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);

	Mode m_mode;
};

}
