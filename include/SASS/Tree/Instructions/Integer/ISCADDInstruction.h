#pragma once

#include "SASS/Tree/Instructions/PredicatedInstruction.h"

#include "SASS/Tree/BinaryUtils.h"
#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/I8Immediate.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {

class ISCADDInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,
		NEG_I = 0x0100000000000000,
		NEG_A = 0x0002000000000000,
		NEG_B = 0x0001000000000000
	};

	SASS_FLAGS_FRIEND()

	ISCADDInstruction(Register *destination, Register *sourceA, Composite *sourceB, I8Immediate *sourceC, Flags flags = Flags::None)
		: m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_flags(flags) {}

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

	const I8Immediate *GetSourceC() const { return m_sourceC; }
	I8Immediate *GetSourceC() { return m_sourceC; }
	void SetSourceC(I8Immediate *sourceC) { m_sourceC = sourceC; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { m_destination };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { m_sourceA, m_sourceB, m_sourceC };
	}

	// Formatting

	std::string OpCode() const override { return "ISCADD"; }

	std::string Operands() const override
	{
		std::string code;

		// Destination
		code += m_destination->ToString();
		code += ", ";

		// SourceA
		if (m_flags & Flags::NEG_A)
		{
			code += "-";
		}
		code += m_sourceA->ToString();
		code += ", ";

		// SourceB
		if (m_flags & Flags::NEG_I || m_flags & Flags::NEG_B)
		{
			code += "-";
		}
		code += m_sourceB->ToString();
		if (m_flags & Flags::NEG_I && m_flags & Flags::NEG_B)
		{
			code += ".NEG";
		}
		code += ", ";

		// SourceC
		code += m_sourceC->ToString();

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCodeComposite(0x5c18000000000000, m_sourceB);
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_flags);
	}

	std::uint64_t BinaryOperands() const override
	{
		return BinaryUtils::OperandRegister0(m_destination) |
		       BinaryUtils::OperandRegister8(m_sourceA) |
		       BinaryUtils::OperandComposite(m_sourceB) |
		       BinaryUtils::OperandLiteral39W8(m_sourceC);
	}

	// Hardware properties

	HardwareClass GetHardwareClass() const override { return HardwareClass::Shift; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Register *m_sourceA = nullptr;
	Composite *m_sourceB = nullptr;
	I8Immediate *m_sourceC = nullptr;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(ISCADDInstruction)

}
