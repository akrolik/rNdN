#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

#include "PTX/Tree/Operands/Operand.h"
#include "PTX/Tree/Operands/Variables/Register.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface(PermuteInstruction)

template<class T>
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

	PermuteInstruction(const Register<T> *destinationD, const TypedOperand<T> *sourceA, const TypedOperand<T> *sourceB, const TypedOperand<T> *sourceC, Mode mode = Mode::Generic) : InstructionBase_3<T>(destinationD, sourceA, sourceB, sourceC), m_mode(mode) {}

	Mode GetMode() const { return m_mode; }
	void SetMode(Mode mode) { m_mode = mode; }

	static std::string Mnemonic() { return "prmt"; }

	std::string OpCode() const override
	{
		return Mnemonic() + T::Name() + ModeString(m_mode);
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);

	Mode m_mode;
};

DispatchImplementation(PermuteInstruction)

}
