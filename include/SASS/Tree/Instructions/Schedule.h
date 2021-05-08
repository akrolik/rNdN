#pragma once

#include <vector>

namespace SASS {

class Schedule
{
public:
	enum class Barrier : std::uint8_t {
		SB0  = 0,
		SB1  = 1,
		SB2  = 2,
		SB3  = 3,
		SB4  = 4,
		SB5  = 5,
		None = 7
	};

	enum class ReuseCache : std::uint8_t {
		OperandA = 1,
		OperandB = 2,
		OperandC = 3
	};

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

	const std::vector<Barrier>& GetWaitBarriers() const { return m_waitBarriers; }
	void SetWaitBarriers(const std::vector<Barrier>& barriers) { m_waitBarriers = barriers; }

	// Register reuse cache

	const std::vector<ReuseCache>& GetReuseCache() const { return m_reuseCache; }
	void SetReuseCache(const std::vector<ReuseCache>& reuseCache) { m_reuseCache = reuseCache; }

	// Format

	std::string OperandModifier(ReuseCache operand) const
	{
		for (auto reuse : m_reuseCache)
		{
			if (reuse == operand)
			{
				return ".reuse";
			}
		}
		return "";
	}

	std::string ToString() const
	{
		std::string code = "[";

		// Reuse cache
		code += "C";
		if (m_reuseCache.size() == 0)
		{
			code += "-";
		}
		else
		{
			for (auto reuse : m_reuseCache)
			{
				code += std::to_string(static_cast<std::uint8_t>(reuse));
			}
		}
		code += ";";

		// Wait barriers
		code += "B";
		if (m_waitBarriers.size() == 0)
		{
			code += "-";
		}
		else
		{
			for (auto wait : m_waitBarriers)
			{
				code += std::to_string(static_cast<std::uint8_t>(wait));
			}
		}
		code += ";";

		// Write barrier

		code += "R";
		if (m_readBarrier == Barrier::None)
		{
			code += "-";
		}
		else
		{
			code += std::to_string(static_cast<std::uint8_t>(m_readBarrier));
		}
		code += ";";

		// Read barrier

		code += "W";
		if (m_writeBarrier == Barrier::None)
		{
			code += "-";
		}
		else
		{
			code += std::to_string(static_cast<std::uint8_t>(m_writeBarrier));
		}
		code += ";";

		// Yield/stall

		code += "Y" + std::to_string(m_yield) + ";";
		code += "S" + std::to_string(m_stall);

		return code + "]";
	}

	std::uint32_t GenCode() const
	{
		std::uint32_t code = 0u;
		for (auto reuse : m_reuseCache)
		{
			code |= ((1 << static_cast<std::uint8_t>(reuse)) << 17);
		}
		for (auto wait : m_waitBarriers)
		{
			code |= ((1 << static_cast<std::uint8_t>(wait))  << 11);
		}
		code |= (static_cast<std::uint8_t>(m_readBarrier)        << 8);
		code |= (static_cast<std::uint8_t>(m_writeBarrier)       << 5);
		code |= (m_yield                                         << 4);
		code |= (m_stall                                         << 0);
		return code;
	}

private:
	std::uint8_t m_stall = 0;
	bool m_yield = false;
	Barrier m_writeBarrier = Barrier::None;
	Barrier m_readBarrier = Barrier::None;
	std::vector<Barrier> m_waitBarriers;
	std::vector<ReuseCache> m_reuseCache;
};

}
