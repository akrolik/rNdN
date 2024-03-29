#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

namespace PTX {

class MemoryBarrierInstruction : public PredicatedInstruction, public DispatchBase
{
public:
	enum class Level {
		CTA,
		GPU,
		System
	};

	static std::string LevelString(Level level)
	{
		switch (level)
		{
			case Level::CTA:
				return ".cta";
			case Level::GPU:
				return ".gl";
			case Level::System:
				return ".sys";
		}
		return ".<unknown>";
	}

	MemoryBarrierInstruction(Level level) : m_level(level) {}

	// Analysis properties

	bool HasSideEffect() const override { return true; }

	// Properties

	Level GetLevel() const { return m_level; }
	void SetLevel(Level level) { m_level = level; }

	// Formatting

	static std::string Mnemonic() { return "membar"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + LevelString(m_level);
	}

	std::vector<const Operand *> GetOperands() const
	{
		return {};
	}

	std::vector<Operand *> GetOperands()
	{
		return {};
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	Level m_level;
};

}
