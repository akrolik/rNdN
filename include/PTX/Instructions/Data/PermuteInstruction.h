#pragma once

#include "PTX/Instructions/InstructionBase.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

class PermuteInstruction : public InstructionBase_3<Bit32Type, Bit32Type, Bit32Type, Bit16Type>
{
public:
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

	PermuteInstruction(const Register<Bit32Type> *destinationD, const TypedOperand<Bit32Type> *sourceA, const TypedOperand<Bit32Type> *sourceB, const TypedOperand<Bit16Type> *sourceC, Mode mode = Mode::Generic) : InstructionBase_3<Bit32Type, Bit32Type, Bit32Type, Bit16Type>(destinationD, sourceA, sourceB, sourceC), m_mode(mode) {}

	Mode GetMode() const { return m_mode; }
	void SetMode(Mode mode) { m_mode = mode; }

	static std::string Mnemonic() { return "prmt"; }

	std::string OpCode() const override
	{
		return Mnemonic() + Bit32Type::Name() + ModeString(m_mode);
	}

private:
	Mode m_mode;
};

}
