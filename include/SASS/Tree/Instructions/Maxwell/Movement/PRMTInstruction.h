#pragma once

#include "SASS/Tree/Instructions/Maxwell/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/BinaryUtils.h"

#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Maxwell {

class PRMTInstruction : public PredicatedInstruction
{
public:
	enum class Mode : std::uint64_t {
		None  = 0,
		F4E   = 0x0001000000000000,
		B4E   = 0x0002000000000000,
		RC8   = 0x0003000000000000,
		ECL   = 0x0004000000000000,
		ECR   = 0x0005000000000000,
		RC16  = 0x0006000000000000
	};

	PRMTInstruction(Register *destination, Register *sourceA, Composite *sourceB, Register *sourceC, Mode mode = Mode::None)
		: m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_mode(mode) {}

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

	std::string OpCode() const override { return "PRMT"; }

	std::string OpModifiers() const override
	{
		std::string code;
		switch (m_mode)
		{
			case Mode::F4E: code += ".F4E"; break;
			case Mode::B4E: code += ".B4E"; break;
			case Mode::RC8: code += ".RC8"; break;
			case Mode::ECL: code += ".ECL"; break;
			case Mode::ECR: code += ".ECR"; break;
			case Mode::RC16: code += ".RC16"; break;
		}
		return code;
	}

	std::string Operands() const override
	{             
		std::string code;

		// Destination
		code += m_destination->ToString();
		code += ", ";

		// SourceA
		code += m_sourceA->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandA);
		code += ", ";

		// SourceB
		code += m_sourceB->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandB);
		code += ", ";

		// SourceC
		code += m_sourceC->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandC);

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return BinaryUtils::OpCodeComposite(0x5bc0000000000000, m_sourceB);
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		auto code = BinaryUtils::OpModifierFlags(m_mode);
		if (m_sourceB->GetOpModifierNegate())
		{
			code |= 0x0100000000000000;
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

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::Integer; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Register *m_sourceA = nullptr;
	Composite *m_sourceB = nullptr;
	Register *m_sourceC = nullptr;

	Mode m_mode = Mode::None;
};

}
}
