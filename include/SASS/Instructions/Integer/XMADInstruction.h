#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/Composite.h"
#include "SASS/Operands/Register.h"

namespace SASS {

class XMADInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0x0,
		NEG_I = 0x0100000000000000,
		CC    = 0x0000800000000000,
		CBCC  = 0x0010000000000000,
		H1_A  = 0x0020000000000000,
		H1_B  = 0x0000000800000000,
	};

	SASS_FLAGS_FRIEND()

	enum class Type1 : std::uint64_t {
		U16 = 0x0000000000000000,
		S16 = 0x0001000000000000
	};

	enum class Type2 : std::uint64_t {
		U16 = 0x0000000000000000,
		S16 = 0x0002000000000000
	};

	enum class Mode : std::uint64_t {
		None    = 0x0,
		X       = 0x0040000000000000,
		MRG     = 0x0000002000000000,
		PSL     = 0x0000001000000000,
		CHI     = 0x0008000000000000,
		CLO     = 0x0004000000000000,
		CSFU    = 0x000c000000000000,
		PSL_CLO = 0x0004001000000000
	};

	XMADInstruction(Register *destination, Register *sourceA, Composite *sourceB, Register *sourceC, Mode mode = Mode::None, Flags flags = Flags::None, Type1 type1 = Type1::U16, Type2 type2 = Type2::U16)
		: PredicatedInstruction({destination, sourceA, sourceB, sourceC}), m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_mode(mode), m_type1(type1), m_type2(type2), m_flags(flags) {}

	// Properties

	const Register *GetDestination() const { return m_destination; }
	Register *GetDestination() { return m_destination; }
	void SetDestination(Register *destination) { m_destination = destination; }

	const Register *GetSourceA() const { return m_sourceA; }
	Register *GetSourceA() { return m_sourceA; }
	void SetSourceA(Register *sourceA) { m_sourceA = sourceA; }

	const Composite *GetSourceB() const { return m_sourceB; }
	Composite *GetSourceB() { return m_sourceB; }
	void SetSourceB(Composite *sourceB) { m_sourceB = sourceB; }

	const Register *GetSourceC() const { return m_sourceC; }
	Register *GetSourceC() { return m_sourceC; }
	void SetSourceC(Register *sourceC) { m_sourceC = sourceC; }

	Mode GetMode() const { return m_mode; }
	void SetMode(Mode mode) { m_mode = mode; }

	Type1 GetType1() const { return m_type1; }
	void SetType1(Type1 type1) { m_type1 = type1; }

	Type2 GetType2() const { return m_type2; }
	void SetType2(Type2 type2) { m_type2 = type2; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Formatting

	std::string OpCode() const override { return "XMAD"; }

	std::string OpModifiers() const override
	{
		std::string code;
		if (m_type1 == Type1::S16)
		{
			code += ".S16";
		}
		if (m_type2 == Type2::S16)
		{
			code += ".S16";
		}
		switch (m_mode)
		{
			case Mode::MRG: code += ".MRG"; break;
			case Mode::PSL: code += ".PSL"; break;
			case Mode::CHI: code += ".CHI"; break;
			case Mode::CLO: code += ".CLO"; break;
			case Mode::CSFU: code += ".CSFU"; break;
			case Mode::PSL_CLO: code += ".PSL_CLO"; break;
		}
		if (m_flags & Flags::CBCC)
		{
			code += ".CBCC";
		}
		return code;
	}

	std::string Operands() const override
	{
		std::string code;

		// Destination
		code += m_destination->ToString();
		if (m_flags & Flags::CC)
		{
			code += ".CC";
		}
		code += ", ";

		// SourceA
		code += m_sourceA->ToString();
		if (m_flags & Flags::H1_A)
		{
			code += ".H1";
		}
		code += ", ";

		// SourceB
		if (m_flags & Flags::NEG_I)
		{
			code += "-";
		}
		code += m_sourceB->ToString();
		if (m_flags & Flags::H1_B)
		{
			code += ".H1";
		}
		code += ", ";

		// SourceC
		code += m_sourceC->ToString();

		return code;
	}

	// Binary

	// XMAD R19, R17, c[0x0] [0x8], R0 ;                  /* 0x4e 000000002 7 11 13 */
	// XMAD.MRG R20, R17, c[0x0] [0x8].H1, RZ ;           /* 0x4f 107f80002 7 11 14 */
	// XMAD.PSL.CBCC R17, R17.H1, R20.H1, R19 ;           /* 0x5b 3009980 14 7 11 11 */

	std::uint64_t BinaryOpCode() const override
	{
		if (dynamic_cast<Constant *>(m_sourceB))
		{
			return 0x4e00000000000000; // Constant
		}
		//TODO: Immediate
		return 0x5b00000000000000; // Register
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		std::uint64_t code = BinaryUtils::OpModifierFlags(m_type1) | BinaryUtils::OpModifierFlags(m_type2);
		if (dynamic_cast<const Constant *>(m_sourceB))
		{
			if (m_flags & Flags::H1_B)
			{
				code |= 0x0010000000000000; // H1B for constant
			}
			code |= m_flags & ~Flags::H1_B;

			// Use different representation for constant mode

			switch (m_mode)
			{
				case Mode::CLO:
					code |= 0x0004000000000000;
					break;
				case Mode::CHI:
					code |= 0x0008000000000000;
					break;
				case Mode::CSFU:
					code |= 0x000c000000000000;
					break;
				case Mode::X:
					code |= 0x0040000000000000;
					break;
				case Mode::PSL:
					code |= 0x0080000000000000;
					break;
				case Mode::MRG:
					code |= 0x0100000000000000;
					break;
				case Mode::PSL_CLO:
					code |= 0x0084000000000000;
					break;
			}
		}
		else
		{
			code |= BinaryUtils::OpModifierFlags(m_flags);
			code |= BinaryUtils::OpModifierFlags(m_mode);
		}
		return code;
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandRegister8(m_sourceA) |
		       BinaryUtils::OperandComposite(m_sourceB) |
		       BinaryUtils::OperandRegister39(m_sourceC);
	}

private:
	Register *m_destination = nullptr;
	Register *m_sourceA = nullptr;
	Composite *m_sourceB = nullptr;
	Register *m_sourceC = nullptr;

	Type1 m_type1 = Type1::U16;
	Type2 m_type2 = Type2::U16;
	Mode m_mode = Mode::None;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(XMADInstruction)

}
