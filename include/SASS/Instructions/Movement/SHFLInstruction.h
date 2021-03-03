#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

#include "SASS/BinaryUtils.h"
#include "SASS/Operands/Composite.h"
#include "SASS/Operands/Predicate.h"
#include "SASS/Operands/Register.h"

namespace SASS {

class SHFLInstruction : public PredicatedInstruction
{
public:
	enum class ShuffleOperator : std::uint64_t {
		IDX  = 0x0000000000000000,
		UP   = 0x0000000040000000,
		DOWN = 0x0000000080000000,
		BFLY = 0x00000000c0000000
	};

	SHFLInstruction(Predicate *destinationA, Register *destinationB, Register *sourceA, Composite *sourceB, Composite *sourceC, ShuffleOperator shuffleOp)
		: PredicatedInstruction({destinationA, destinationB, sourceA, sourceB, sourceC}), m_destinationA(destinationA), m_destinationB(destinationB), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_shuffleOp(shuffleOp) {}

	// Properties

	const Predicate *GetDestinationA() const { return m_destinationA; }
	Predicate *GetDestinationA() { return m_destinationA; }
	void SetDestinationA(Predicate *destinationA) { m_destinationA = destinationA; }

	const Register *GetDestinationB() const { return m_destinationB; }
	Register *GetDestinationB() { return m_destinationB; }
	void SetDestinationB(Register *destinationB) { m_destinationB = destinationB; }

	const Register *GetSourceA() const { return m_sourceA; }
	Register *GetSourceA() { return m_sourceA; }
	void SetSourceA(Register *sourceA) { m_sourceA = sourceA; }

	const Composite *GetSourceB() const { return m_sourceB; }
	Composite *GetSourceB() { return m_sourceB; }
	void SetSourceB(Composite *sourceB) { m_sourceB = sourceB; }

	const Composite *GetSourceC() const { return m_sourceC; }
	Composite *GetSourceC() { return m_sourceC; }
	void SetSourceC(Composite *sourceC) { m_sourceC = sourceC; }

	ShuffleOperator GetShuffleOp() const { return m_shuffleOp; }
	void SetShuffleOp(ShuffleOperator shuffleOp) { m_shuffleOp = shuffleOp; }

	// Formatting
	
	std::string OpCode() const override { return "SHFL"; }

	std::string OpModifiers() const override
	{
		switch (m_shuffleOp)
		{
			case ShuffleOperator::IDX: return ".IDX";
			case ShuffleOperator::UP: return ".UP";
			case ShuffleOperator::DOWN: return ".DOWN";
			case ShuffleOperator::BFLY: return ".BFLY";
		}
		return "";
	}

	std::string Operands() const override
	{
		std::string code;
		
		// Destination
		code += m_destinationA->ToString();
		code += ", ";
		code += m_destinationB->ToString();
		code += ", ";

		// SourceA
		code += m_sourceA->ToString();
		code += ", ";

		// SourceB
		code += m_sourceB->ToString();
		code += ", ";

		// SourceC
		code += m_sourceC->ToString();

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xef10000000000000;
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		std::uint64_t code = 0;
		
		// Operation
		code |= BinaryUtils::OpModifierFlags(m_shuffleOp);

		// Use 8-bit integer for source B
		if (dynamic_cast<const I32Immediate *>(m_sourceB))
		{
			code |= 0x0000000010000000;
		}
		// Use 13-bit integer for source C
		if (dynamic_cast<const I32Immediate *>(m_sourceC))
		{
			code |= 0x0000000020000000;
		}

		return code;
	}

	std::uint64_t BinaryOperands() const override
	{
		std::uint64_t code = 0x0;
		code |= BinaryUtils::OperandPredicate48(m_destinationA);
		code |= BinaryUtils::OperandRegister0(m_destinationB);
		code |= BinaryUtils::OperandRegister8(m_sourceA);

		if (auto immediateB = dynamic_cast<const I32Immediate *>(m_sourceB))
		{
			code |= BinaryUtils::OperandLiteral20W8(immediateB);
		}
		else if (auto registerB = dynamic_cast<const Register *>(m_sourceB))
		{
			code |= BinaryUtils::OperandRegister20(registerB);
		}

		if (auto immediateC = dynamic_cast<const I32Immediate *>(m_sourceC))
		{
			code |= BinaryUtils::OperandLiteral34W13(immediateC);
		}
		else if (auto registerC = dynamic_cast<const Register *>(m_sourceC))
		{
			code |= BinaryUtils::OperandRegister39(registerC);
		}

		return code;
	}

private:
	Predicate *m_destinationA = nullptr;
	Register *m_destinationB = nullptr;
	Register *m_sourceA = nullptr;
	Composite *m_sourceB = nullptr;
	Composite *m_sourceC = nullptr;

	ShuffleOperator m_shuffleOp;
};

}
