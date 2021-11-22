#pragma once

#include "SASS/Tree/Instructions/Volta/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/I8Immediate.h"

#include <vector>

namespace SASS {
namespace Volta {

class DEPBARInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None = 0,
		LE   = (1 << 0)
	};

	enum class Barrier : std::uint64_t {
		SB0 = 0x0,
		SB1 = 0x1,
		SB2 = 0x2,
		SB3 = 0x3,
		SB4 = 0x4,
		SB5 = 0x5
	};

	SASS_FLAGS_FRIEND()

	DEPBARInstruction(Barrier barrier, I8Immediate *value, const std::vector<Barrier>& barriers, Flags flags = Flags::None)
		: m_barrier(barrier), m_value(value), m_barriers(barriers), m_flags(flags) {}

	DEPBARInstruction(Barrier barrier, I8Immediate *value, Flags flags = Flags::None)
		: DEPBARInstruction(barrier, value, {}, flags) {}

	DEPBARInstruction(Barrier barrier, const std::vector<Barrier>& barriers, Flags flags = Flags::None)
		: DEPBARInstruction(barrier, nullptr, barriers, flags) {}

	// Properties

	Barrier GetBarrier() const { return m_barrier; }
	void SetBarrier(Barrier barrier) { m_barrier = barrier; }

	const I8Immediate *GetValue() const { return m_value; }
	I8Immediate *GetValue() { return m_value; }
	void SetValue(I8Immediate *value) { m_value = value; }

	const std::vector<Barrier>& GetBarriers() const { return m_barriers; }
	std::vector<Barrier>& GetBarriers() { return m_barriers; }
	void SetBarriers(const std::vector<Barrier>& barriers) { m_barriers = barriers; }

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { m_value };
	}

	// Formatting

	std::string OpCode() const override { return "DEPBAR"; }

	std::string OpModifiers() const override
	{
		if (m_flags & Flags::LE)
		{
			return ".LE";
		}
		return "";
	}

	std::string Operands() const override
	{
		std::string code;

		// Barrier
		switch (m_barrier)
		{
			case Barrier::SB0: code += "SB0"; break;
			case Barrier::SB1: code += "SB1"; break;
			case Barrier::SB2: code += "SB2"; break;
			case Barrier::SB3: code += "SB3"; break;
			case Barrier::SB4: code += "SB4"; break;
			case Barrier::SB5: code += "SB5"; break;
		}

		// Literal
		if (m_value != nullptr)
		{
			code += ", ";
			code += m_value->ToString();
		}

		// Barriers
		if (m_barriers.size() > 0)
		{
			code += ", {";
			for (auto barrier : m_barriers)
			{
				code += std::to_string(static_cast<std::underlying_type<Barrier>::type>(barrier));
			}
			code += "}";
		}

		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x91a;
	}

	std::uint64_t ToBinary() const override
	{
		auto code = PredicatedInstruction::ToBinary();

		// Literal
		if (m_value != nullptr)
		{
			code |= BinaryUtils::Format(m_value->ToBinary(), 38, 0x3f);
		}

		// Barriers
		if (m_barriers.size() > 0)
		{
			for (auto barrier : m_barriers)
			{
				code |= BinaryUtils::FlagBit(true, static_cast<std::underlying_type<Barrier>::type>(barrier) + 32);
			}
		}

		// Barrier
		code |= BinaryUtils::Format(m_barrier, 44, 0x7);

		// LE flag
		code |= BinaryUtils::FlagBit(m_flags & Flags::LE, 47);

		return  code;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::Control; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Barrier m_barrier;
	I8Immediate *m_value = nullptr;
	Flags m_flags = Flags::None;
	std::vector<Barrier> m_barriers;
};

SASS_FLAGS_INLINE(DEPBARInstruction)

}
}
