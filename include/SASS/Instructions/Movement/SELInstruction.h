#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/Composite.h"
#include "SASS/Operands/Predicate.h"
#include "SASS/Operands/Register.h"

namespace SASS {

class SELInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,                    
		NEG_I = 0x0100000000000000,
		NOT_C = 0x0000040000000000
	};

	SASS_FLAGS_FRIEND()

	SELInstruction(Register *destination, Register *sourceA, Composite *sourceB, Predicate *sourceC, Flags flags = Flags::None)
		: PredicatedInstruction({destination, sourceA, sourceB, sourceC}), m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_flags(flags) {}

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

	const Predicate *GetSourceC() const { return m_sourceC; }
	Predicate *GetSourceC() { return m_sourceC; }
	void SetSourceB(Predicate *sourceC) { m_sourceC = sourceC; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Formatting
	
	std::string OpCode() const override { return "SEL"; }

	std::string Operands() const override
	{
		std::string code;
		
		// Destination
		code += m_destination->ToString();
		code += ", ";

		// SourceA
		code += m_sourceA->ToString();
		code += ", ";

		// SourceB
		if (m_flags & Flags::NEG_I)
		{
			code += "-";
		}
		code += m_sourceB->ToString();
		code += ", ";

		// SourceC
		if (m_flags & Flags::NOT_C)
		{
			code += "!";
		}
		code += m_sourceC->ToString();

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCodeComposite(0x5ca0000000000000, m_sourceB);
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
		       BinaryUtils::OperandPredicate39(m_sourceC);
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Register *m_sourceA = nullptr;
	Composite *m_sourceB = nullptr;
	Predicate *m_sourceC = nullptr;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(SELInstruction)

}
