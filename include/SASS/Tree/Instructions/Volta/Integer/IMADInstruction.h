#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Register.h"
#include "SASS/Tree/Operands/Predicate.h"

namespace SASS {
namespace Volta {

class IMADInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0,
		X     = (1 << 0),
		U32   = (1 << 1),
		NEG_C = (1 << 2),
		INV_C = (1 << 3),
		NOT_D = (1 << 4),
	};

	SASS_FLAGS_FRIEND()

	enum class Mode : std::uint64_t {
		Default = 0x4,
		WIDE    = 0x5,
		HI      = 0x7
	};

	// Full
	IMADInstruction(Register *destinationA, Predicate *destinationB, Register *sourceA, Composite *sourceB, Composite *sourceC, Predicate *sourceD, Mode mode = Mode::Default, Flags flags = Flags::None)
		: m_destinationA(destinationA), m_destinationB(destinationB), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_sourceD(sourceD), m_mode(mode), m_flags(flags) {}

	// MAD
	IMADInstruction(Register *destinationA, Register *sourceA, Composite *sourceB, Composite *sourceC, Mode mode = Mode::Default, Flags flags = Flags::None)
		: IMADInstruction(destinationA, nullptr, sourceA, sourceB, sourceC, nullptr, mode, flags) {}

	// Carry-out
	IMADInstruction(Register *destinationA, Predicate *destinationB, Register *sourceA, Composite *sourceB, Composite *sourceC, Mode mode = Mode::Default, Flags flags = Flags::None)
		: IMADInstruction(destinationA, destinationB, sourceA, sourceB, sourceC, nullptr, mode, flags) {}

	// Carry-in
	IMADInstruction(Register *destinationA, Register *sourceA, Composite *sourceB, Composite *sourceC, Predicate *sourceD, Mode mode = Mode::Default, Flags flags = Flags::None)
		: IMADInstruction(destinationA, nullptr, sourceA, sourceB, sourceC, sourceD, mode, flags) {}

	// Properties

	const Register *GetDestinationA() const { return m_destinationA; }
	Register *GetDestinationA() { return m_destinationA; }
	void SetDestinationA(Register *destinationA) { m_destinationA = destinationA; }

	const Predicate *GetDestinationB() const { return m_destinationB; }
	Predicate *GetDestinationB() { return m_destinationB; }
	void SetDestinationB(Predicate *destinationB) { m_destinationB = destinationB; }

	const Register *GetSourceA() const { return m_sourceA; }
	Register *GetSourceA() { return m_sourceA; }
	void SetSourceA(Register *sourceA) { m_sourceA = sourceA; }

	const Composite *GetSourceB() const { return m_sourceB; }
	Composite *GetSourceB() { return m_sourceB; }
	void SetSourceB(Composite *sourceB) { m_sourceB = sourceB; }

	const Composite *GetSourceC() const { return m_sourceC; }
	Composite *GetSourceC() { return m_sourceC; }
	void SetSourceC(Composite *sourceC) { m_sourceC = sourceC; }

	const Predicate *GetSourceD() const { return m_sourceD; }
	Predicate *GetSourceD() { return m_sourceD; }
	void SetSourceD(Predicate *sourceD) { m_sourceD = sourceD; }

	Mode GetMode() const { return m_mode; }
	void SetMode(Mode mode) { m_mode = mode; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { m_destinationA, m_destinationB };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { m_sourceA, m_sourceB, m_sourceC, m_sourceD };
	}

	// Formatting

	std::string OpCode() const override { return "IMAD"; }

	std::string OpModifiers() const override
	{
		std::string code;
		switch (m_mode)
		{
			case Mode::WIDE: code += ".WIDE"; break;
			case Mode::HI: code += ".HI"; break;
			case Mode::Default:
			{
				if (m_sourceA == RZ || m_sourceB == RZ)
				{
					code += ".MOV";
				}
				else if (m_sourceB->GetKind() == Operand::Kind::Immediate)
				{
					auto value = static_cast<I32Immediate *>(m_sourceB)->GetValue();
					if (value == 0)
					{
						code += ".MOV";
					}
					else if (value == 1)
					{
						code += ".ADD";
					}
					else if (value == 2 && m_sourceC == RZ)
					{
						code += ".SHL";
					}
				}
			}
		}
		if (m_flags & Flags::U32)
		{
			code += ".U32";
		}
		if (m_flags & Flags::X)
		{
			code += ".X";
		}
		return code;
	}

	std::string Operands() const override
	{             
		std::string code;

		// DestinationA
		code += m_destinationA->ToString();
		code += ", ";

		// DestinationB
		if (m_destinationB != nullptr)
		{
			code += m_destinationB->ToString();
			code += ", ";
		}

		// SourceA
		code += m_sourceA->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandA);
		code += ", ";

		// SourceB
		code += m_sourceB->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandB);
		code += ", ";

		// SourceC
		if (m_flags & Flags::NEG_C)
		{
			code += "-";
		}

		if (m_flags & Flags::X && m_flags & Flags::INV_C)
		{
			code += "~";
		}
		code += m_sourceC->ToString();
		code += m_schedule.OperandModifier(Schedule::ReuseCache::OperandC);

		// SourceD
		if (m_sourceD != nullptr)
		{
			code += ", ";
			if (m_flags & Flags::NOT_D)
			{
				code += "!";
			}
			code += m_sourceD->ToString();
		}

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		auto mode = static_cast<std::underlying_type<Mode>::type>(m_mode);
		if (m_sourceB->GetKind() == Operand::Kind::Immediate)
		{
			return (0x820 | mode);
		}
		else if (m_sourceB->GetKind() == Operand::Kind::Constant)
		{
			return (0xa20 | mode);
		}
		else if (m_sourceC->GetKind() == Operand::Kind::Immediate)
		{
			return (0x420 | mode);
		}
		else if (m_sourceC->GetKind() == Operand::Kind::Constant)
		{
			return (0x620 | mode);
		}
		return (0x220 | mode);
	}

	std::uint64_t ToBinary() const override
	{
		auto code = PredicatedInstruction::ToBinary();

		// DestinationA
		code |= BinaryUtils::OperandRegister16(m_destinationA);

		// SourceA
		code |= BinaryUtils::OperandRegister24(m_sourceA);

		// SourceB/SourceC (immediate/constant)
		if (m_sourceC->GetKind() == Operand::Kind::Register)
		{
			code |= BinaryUtils::OperandComposite(m_sourceB);
		}
		else
		{
			code |= BinaryUtils::OperandComposite(m_sourceC, m_flags & NEG_C);
		}

		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		auto code = PredicatedInstruction::ToBinaryHi();

		// SourceC (register)/SourceB (if SourceC not register)
		if (m_sourceC->GetKind() == Operand::Kind::Register)
		{
			auto registerC = static_cast<Register *>(m_sourceC);
			code |= BinaryUtils::OperandRegister0(registerC);
		}
		else
		{
			auto registerB = static_cast<Register *>(m_sourceB);
			code |= BinaryUtils::OperandRegister0(registerB);
		}

		// DestinationB
		code |= BinaryUtils::OperandPredicate17(m_destinationB);

		// Flags
		code |= BinaryUtils::FlagBit(!(m_flags & Flags::U32), 9);
		code |= BinaryUtils::FlagBit(m_flags & Flags::X, 10);

		if (m_sourceC->GetKind() != Operand::Kind::Immediate)
		{
			code |= BinaryUtils::FlagBit(m_flags & Flags::NEG_C || m_flags & Flags::INV_C, 11);
		}

		// SourceD
		code |= BinaryUtils::OperandPredicate23(m_sourceD, m_flags & Flags::NOT_D);

		return code;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::Integer; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Register *m_destinationA = nullptr;
	Predicate *m_destinationB = nullptr;

	Register *m_sourceA = nullptr;
	Composite *m_sourceB = nullptr;
	Composite *m_sourceC = nullptr;
	Predicate *m_sourceD = nullptr;

	Mode m_mode = Mode::Default;
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(IMADInstruction)

}
}
