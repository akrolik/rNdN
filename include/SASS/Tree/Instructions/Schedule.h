#pragma once

#include "SASS/Tree/Instructions/BinaryUtils.h"

#include "Libraries/robin_hood.h"

namespace SASS {

class Schedule
{
public:
	enum class Barrier : std::uint8_t {
		None = 0,
		SB0  = (1 << 0),
		SB1  = (1 << 1),
		SB2  = (1 << 2),
		SB3  = (1 << 3),
		SB4  = (1 << 4),
		SB5  = (1 << 5)
	};

	enum class ReuseCache : std::uint8_t {
		None     = 0,
		OperandA = (1 << 0),
		OperandB = (1 << 1),
		OperandC = (1 << 2)
	};

	static std::string BarrierString(Barrier barrier)
	{
		if (barrier == Barrier::None)
		{
			return "-";
		}

		std::string code;
		if (!!(barrier & Barrier::SB0))
		{
			code += "0";
		}
		if (!!(barrier & Barrier::SB1))
		{
			code += "1";
		}
		if (!!(barrier & Barrier::SB2))
		{
			code += "2";
		}
		if (!!(barrier & Barrier::SB3))
		{
			code += "3";
		}
		if (!!(barrier & Barrier::SB4))
		{
			code += "4";
		}
		if (!!(barrier & Barrier::SB5))
		{
			code += "5";
		}
		return code;
	}

	static std::uint8_t BarrierIndex(Barrier barrier)
	{
		auto logValue = __builtin_ctz(static_cast<std::underlying_type_t<Barrier>>(barrier));
		return static_cast<std::uint8_t>(logValue);
	}

	static Barrier BarrierFromIndex(std::uint8_t index)
	{
		return static_cast<Barrier>(1 << index);
	}

	static std::string ReuseCacheString(ReuseCache cache)
	{
		if (cache == ReuseCache::None)
		{
			return "-";
		}

		std::string code;
		if (!!(cache & ReuseCache::OperandA))
		{
			code += "0";
		}
		if (!!(cache & ReuseCache::OperandB))
		{
			code += "1";
		}
		if (!!(cache & ReuseCache::OperandC))
		{
			code += "2";
		}
		return code;
	}

	static std::uint8_t ReuseCacheIndex(ReuseCache cache)
	{
		auto logValue = __builtin_ctz(static_cast<std::underlying_type_t<ReuseCache>>(cache));
		return static_cast<std::uint8_t>(logValue);
	}

	static ReuseCache ReuseCacheFromIndex(std::uint8_t index)
	{
		return static_cast<ReuseCache>(1 << index);
	}

	SASS_ENUM_FRIEND(Barrier)
	SASS_ENUM_FRIEND(ReuseCache)

	static constexpr std::uint8_t DualIssue = 0;

	// Stall

	std::uint8_t GetStall() const { return m_stall; }
	void SetStall(std::uint8_t stall) { m_stall = stall; }

	// Yield
	
	bool GetYield() const { return m_yield; }
	void SetYield(bool yield) { m_yield = yield; }

	// Barrier

	Barrier GetWriteBarrier() const { return m_writeBarrier; }
	void SetWriteBarrier(Barrier barrier) { m_writeBarrier = barrier; }

	Barrier GetReadBarrier() const { return m_readBarrier; }
	void SetReadBarrier(Barrier barrier) { m_readBarrier = barrier; }

	Barrier GetWaitBarriers() const { return m_waitBarriers; }
	Barrier& GetWaitBarriers() { return m_waitBarriers; }
	void SetWaitBarriers(Barrier barriers) { m_waitBarriers = barriers; }

	// Register reuse cache

	ReuseCache GetReuseCache() const { return m_reuseCache; }
	ReuseCache& GetReuseCache() { return m_reuseCache; }
	void SetReuseCache(ReuseCache reuseCache) { m_reuseCache = reuseCache; }

	// Format

	std::string OperandModifier(ReuseCache operand) const
	{
		if (!!(m_reuseCache & operand))
		{
			return ".reuse";
		}
		return "";
	}

	std::string ToString() const
	{
		std::string code = "[";

		// Reuse cache
		code += "C" + ReuseCacheString(m_reuseCache) + ";";

		// Wait barriers
		code += "B" + BarrierString(m_waitBarriers) + ";";

		// Write barrier

		code += "R" + BarrierString(m_readBarrier) + ";";

		// Read barrier

		code += "W" + BarrierString(m_writeBarrier) + ";";

		// Yield/stall

		code += "Y" + std::to_string(m_yield) + ";";
		code += "S" + std::to_string(m_stall);

		return code + "]";
	}

	std::uint32_t ToBinary() const
	{
		std::uint32_t code = 0u;
		code |= (static_cast<std::uint8_t>(m_reuseCache)   << 17);
		code |= (static_cast<std::uint8_t>(m_waitBarriers) << 11);

		if (m_readBarrier == Barrier::None)
		{
			code |= (7 << 8);
		}
		else
		{
			code |= (BarrierIndex(m_readBarrier) << 8);
		}
		if (m_writeBarrier == Barrier::None)
		{
			code |= (7 << 5);
		}
		else
		{
			code |= (BarrierIndex(m_writeBarrier) << 5);
		}

		code |= (m_yield << 4);
		code |= (m_stall << 0);
		return code;
	}

private:
	std::uint8_t m_stall = 0;
	bool m_yield = false;
	Barrier m_writeBarrier = Barrier::None;
	Barrier m_readBarrier = Barrier::None;
	Barrier m_waitBarriers = Barrier::None;
	ReuseCache m_reuseCache = ReuseCache::None;
};

SASS_ENUM_INLINE(Schedule, Barrier)
SASS_ENUM_INLINE(Schedule, ReuseCache)

}
