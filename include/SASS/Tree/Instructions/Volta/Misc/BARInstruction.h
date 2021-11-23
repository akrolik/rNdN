#pragma once

#include "SASS/Tree/Instructions/Volta/Control/ControlInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Predicate.h"

namespace SASS {
namespace Volta {

class BARInstruction : public ControlInstruction
{
public:
	enum Flags : std::uint64_t {
		None  = 0x0,
		DEFER = (1 << 0) // DEFER_BLOCKING: Valid for .SYNC, .SYNCALL, .RED modes
	};

	SASS_FLAGS_FRIEND()

	enum class Mode : std::uint64_t {
		SYNC     = 0x0,
		ARV      = 0x1,
		RED      = 0x2,
		SCAN     = 0x3,
		SYNCALL  = 0x4
	};

	enum class Reduction : std::uint64_t {
		POPC = 0x0,
		OR   = 0x1,
		AND  = 0x2
	};

	BARInstruction(Flags flags = Flags::None) : m_mode(Mode::SYNCALL), m_flags(flags) {}

	// Note: Both registers must be the same (bits 0..3 for barrier, bits 16..27 for threads)

	BARInstruction(Mode mode, Composite *barrier, Composite *threads = nullptr, Flags flags = Flags::None)
		: m_mode(mode), m_barrier(barrier), m_threads(threads), m_flags(flags) {}

	BARInstruction(Composite *barrier, Composite *threads, Predicate *predicate, bool predicateNegate = false, Flags flags = Flags::None)
		: ControlInstruction(predicate, predicateNegate), m_mode(Mode::SCAN), m_barrier(barrier), m_threads(threads), m_flags(flags) {}

	BARInstruction(Reduction reduction, Composite *barrier, Composite *threads, Predicate *predicate, bool predicateNegate = false, Flags flags = Flags::None)
		: ControlInstruction(predicate, predicateNegate), m_mode(Mode::RED), m_reduction(reduction), m_barrier(barrier), m_threads(threads), m_flags(flags) {}

	// Properties

	Mode GetMode() const { return m_mode; }
	void SetMode(Mode mode) { m_mode = mode; }

	const Composite *GetBarrier() const { return m_barrier; }
	Composite *GetBarrier() { return m_barrier; }
	void SetBarrier(Composite *barrier) { m_barrier = barrier; }

	const Composite *GetThreads() const { return m_threads; }
	Composite *GetThreads() { return m_threads; }
	void SetThreads(Composite *threads) { m_threads = threads; }

	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { m_barrier, m_threads, m_controlPredicate };
	}

	// Formatting

	std::string OpCode() const override { return "BAR"; }

	std::string OpModifiers() const override
	{
		std::string code;
		switch (m_mode)
		{
			case Mode::SYNC: code += ".SYNC"; break;
			case Mode::ARV: code += ".ARV"; break;
			case Mode::RED:
			{
				code += ".RED";
				switch (m_reduction)
				{
					case Reduction::POPC: code += ".POPC"; break;
					case Reduction::AND: code += ".AND"; break;
					case Reduction::OR: code += ".OR"; break;
				}
				break;
			}
			case Mode::SCAN: code += ".SCAN"; break;
			case Mode::SYNCALL: code += ".SYNCALL"; break;
		}
		if (m_flags & Flags::DEFER)
		{
			code += ".DEFER_BLOCKING";
		}
		return code;
	}

	std::string Operands() const override
	{
		std::string code;

		// Barrier
		if (m_barrier != nullptr)
		{
			code += m_barrier->ToString();
		}

		// Threads
		if (m_threads != nullptr)
		{
			code += ", ";
			code += m_threads->ToString();
		}

		// Predicate
		auto predicate = ControlInstruction::Operands();
		if (predicate.size() > 0)
		{
			if (code.size() > 0)
			{
				code += ", ";
			}
			code += predicate;
		}
		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		if (m_barrier == nullptr || m_barrier->GetKind() == Operand::Kind::Immediate)
		{
			if (m_threads == nullptr || m_threads->GetKind() == Operand::Kind::Immediate)
			{
				return 0xb1d; // Barrier = Immediate; Threads = Immediate
			}
			return 0x91d; // Barrier = Immediate; Threads = Register
		}
		else if (m_threads == nullptr || m_threads->GetKind() == Operand::Kind::Immediate)
		{
			return 0x51d; // Barier = Register; Threads = Immediate
		}
		return 0x313; // Barrier = Register; Threads = Register;
	}

	std::uint64_t ToBinary() const override
	{
		auto code = ControlInstruction::ToBinary();

		// Barrier
		if (m_barrier != nullptr)
		{
			code |= BinaryUtils::Format(m_barrier->ToBinary(), 54, 0xf);
		}

		// Threads
		if (m_threads != nullptr)
		{
			code |= BinaryUtils::Format(m_threads->ToBinary(), 42, 0xfff);
		}

		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		auto code = ControlInstruction::ToBinaryHi();

		// Reduction mode
		if (m_mode == Mode::RED)
		{
			code |= BinaryUtils::Format(m_reduction, 10, 0x3);
		}

		// Mode
		code |= BinaryUtils::Format(m_mode, 13, 0x7);

		// Defer flag
		code |= BinaryUtils::FlagBit(m_flags & Flags::DEFER, 16);

		return code;
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::Control; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Mode m_mode;
	Reduction m_reduction;

	Composite *m_barrier = nullptr;
	Composite *m_threads = nullptr;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(BARInstruction)

}
}
