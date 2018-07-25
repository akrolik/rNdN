#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

namespace PTX {

class MemoryBarrierInstruction : public PredicatedInstruction
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

	Level GetLevel() const { return m_level; }
	void SetLevel(Level level) { m_level = level; }

	static std::string Mnemonic() { return "membar"; }

	std::string OpCode() const override
	{
		return Mnemonic() + LevelString(m_level);
	}

protected:
	Level m_level;
};

}
