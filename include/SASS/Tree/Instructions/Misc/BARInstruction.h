#pragma once

#include "SASS/Tree/Instructions/PredicatedInstruction.h"

#include "SASS/Tree/BinaryUtils.h"
#include "SASS/Tree/Operands/Composite.h"
#include "SASS/Tree/Operands/Predicate.h"

namespace SASS {

class BARInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None = 0x0,
		NOT  = 0x0000040000000000
	};

	SASS_FLAGS_FRIEND()

	enum class Mode : std::uint64_t {
		SYNC = 0x0000000000000000,
		ARV  = 0x0000000100000000,
		RED  = 0x0000000200000000
	};

	//TODO: SASS BAR instruction RED destination
	enum class Reduction : std::uint64_t {
		POPC = 0x0000000000000000,
		AND  = 0x0000000800000000,
		OR   = 0x0000001000000000
	};

	BARInstruction(Mode mode, Composite *barrier, Composite *threads = nullptr)
		: m_mode(mode), m_barrier(barrier), m_threads(threads) {}

	BARInstruction(Reduction, Composite *barrier, Composite *threads, Predicate *predicate, Flags flags = Flags::None)
		: m_mode(Mode::RED), m_barrier(barrier), m_threads(threads), m_predicate(predicate), m_flags(flags) {}

	// Properties

	Mode GetMode() const { return m_mode; }
	void SetMode(Mode mode) { m_mode = mode; }

	const Composite *GetBarrier() const { return m_barrier; }
	Composite *GetBarrier() { return m_barrier; }
	void SetBarrier(Composite *barrier) { m_barrier = barrier; }

	const Composite *GetThreads() const { return m_threads; }
	Composite *GetThreads() { return m_threads; }
	void SetThreads(Composite *threads) { m_threads = threads; }

	const Predicate *GetPredicate() const { return m_predicate; }
	Predicate *GetPredicate() { return m_predicate; }
	void SetPredicate(Predicate *predicate) { m_predicate = predicate; }

	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { m_barrier, m_threads, m_predicate };
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
		}
		return code;
	}

	std::string Operands() const override
	{
		std::string code;

		// Barrier
		code += m_barrier->ToString();

		// Threads
		if (m_threads != nullptr)
		{
			code += ", ";
			code += m_threads->ToString();
		}

		if (m_mode == Mode::RED)
		{
			// Predicate
			if (m_predicate != nullptr)
			{
				code += ", ";
				code += m_predicate->ToString();
			}
		}
		return code;
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xf0a8000000000000;
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		std::uint64_t code = BinaryUtils::OpModifierFlags(m_mode) | BinaryUtils::OpModifierFlags(m_flags);
		if (m_mode == Mode::RED)
		{
			code |= BinaryUtils::OpModifierFlags(m_reduction);
		}
		// Use 4-bit integer for barrier
		if (dynamic_cast<I32Immediate *>(m_barrier))
		{
			code |= 0x0000100000000000;
		}
		// Use 12-bit integer for threads
		if (m_threads == nullptr)//TODO:dynamic_cast<I32Immediate *>(m_threads))
		{
			code |= 0x0000080000000000;
		}
		// No predicate register
		if (m_predicate == nullptr)
		{
			code |= 0x0000038000000000;
		}
		return code;
	}

	std::uint64_t BinaryOperands() const override
	{
		std::uint64_t code = 0x0;

		if (auto immediateBarrier = dynamic_cast<I32Immediate *>(m_barrier))
		{
			code |= BinaryUtils::OperandLiteral8W4(immediateBarrier);
		}
		else if (auto registerBarrier = dynamic_cast<Register *>(m_barrier))
		{
			code |= BinaryUtils::OperandRegister8(registerBarrier);
		}

		if (m_threads != nullptr)
		{
			if (auto immediateThreads = dynamic_cast<I32Immediate *>(m_threads))
			{
				code |= BinaryUtils::OperandLiteral20W12(immediateThreads);
			}
			else if (auto registerThreads = dynamic_cast<Register *>(m_threads))
			{
				code |= BinaryUtils::OperandRegister20(registerThreads);
			}
		}

		if (m_mode == Mode::RED)
		{
			if (m_predicate != nullptr)
			{
				code |= BinaryUtils::OperandPredicate39(m_predicate);
			}
		}

		return code;
	}

	// Hardware properties

	// HardwareClass GetHardwareClass() const override { return HardwareClass::GlobalMemory; }
	HardwareClass GetHardwareClass() const override { return HardwareClass::Core; } //TODO:

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Mode m_mode;
	Reduction m_reduction;

	Composite *m_barrier = nullptr;
	Composite *m_threads = nullptr;
	Predicate *m_predicate = nullptr;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(BARInstruction)

}
