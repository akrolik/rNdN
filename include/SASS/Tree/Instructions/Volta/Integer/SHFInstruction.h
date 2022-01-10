#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Register.h"

namespace SASS {
namespace Volta {

class SHFInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,
		HI    = (1 << 0),
		W     = (1 << 1)
	};

	enum class Direction : std::uint64_t {
		L = 0x0,
		R = 0x1
	};

	enum class Type : std::uint64_t {
		S64 = 0x0,
		U64 = 0x1,
		S32 = 0x2,
		U32 = 0x3
	};

	SASS_FLAGS_FRIEND()

	SHFInstruction(Register *destination, Register *sourceA, Composite *sourceB, Register *sourceC, Direction direction, Type type, Flags flags = Flags::None)
		: m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_direction(direction), m_type(type), m_flags(flags) {}

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

	std::string OpCode() const override { return "SHF"; }

	std::string OpModifiers() const override
	{
		std::string code;
		switch (m_direction)
		{
			case Direction::L: code += ".L"; break;
			case Direction::R: code += ".R"; break;
		}
		if (m_flags & Flags::W)
		{
			code += ".W";
		}
		switch (m_type)
		{
			case Type::S64: code += ".S64"; break;
			case Type::U64: code += ".U64"; break;
			case Type::S32: code += ".S32"; break;
			case Type::U32: code += ".U32"; break;
		}
		if (m_flags & Flags::HI)
		{
			code += ".HI";
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

		// SourceC
		if (m_sourceC != nullptr)
		{
			code += ", ";
			code += m_sourceC->ToString();
			code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandC);
		}

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		if (m_sourceC != nullptr)
		{
			return BinaryUtils::OpCode(m_sourceB,
				0x219, // Register
				0x819, // Immediate
				0xa19  // Constant
			);
		}
		return BinaryUtils::OpCode(m_sourceB,
			0x219, // Register
			0x419, // Immediate
			0x619  // Constant
		);
	}

	std::uint64_t ToBinary() const override
	{
		auto code = PredicatedInstruction::ToBinary();

		// Destination
		code |= BinaryUtils::OperandRegister16(m_destination);

		// SourceA
		code |= BinaryUtils::OperandRegister24(m_sourceA);

		// SourceB
		code |= BinaryUtils::OperandComposite(m_sourceB);

		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		auto code = PredicatedInstruction::ToBinaryHi();

		// SourceC
		if (m_sourceC != nullptr)
		{
			code |= BinaryUtils::OperandRegister0(m_sourceC);
		}

		// Modifiers
		code |= BinaryUtils::Format(m_type, 9, 0x3);
		code |= BinaryUtils::Format(m_direction, 12, 0x1);

		// Flags
		code |= BinaryUtils::FlagBit(m_flags & Flags::W, 11);
		code |= BinaryUtils::FlagBit(m_flags & Flags::HI, 16);

		return code;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::Shift; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destination = nullptr;
	Register *m_sourceA = nullptr;
	Composite *m_sourceB = nullptr;
	Register *m_sourceC = nullptr;

	Direction m_direction;
	Type m_type;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(SHFInstruction)

}
}
