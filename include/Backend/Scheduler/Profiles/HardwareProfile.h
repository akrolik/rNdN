#pragma once

#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Scheduler {

class HardwareProfile
{
public:
	enum class FunctionalUnit : std::uint8_t {
		S2R,
		Core,
		HalfCore, // Half throughput
		DoublePrecision,
		SpecialFunction,
		LoadStore
	};

	// Convenience conversions for array storage

	constexpr static unsigned int UNIT_COUNT = 6;
	constexpr static unsigned int BARRIER_COUNT = 6;

	static std::uint8_t FunctionalUnitIndex(FunctionalUnit unit)
	{
		return static_cast<std::underlying_type_t<FunctionalUnit>>(unit);
	}

	static FunctionalUnit FunctionalUnitFromIndex(std::uint8_t index)
	{
		return static_cast<FunctionalUnit>(index);
	}

	// Compute capability

	virtual unsigned int GetComputeCapability() const = 0;

	// Profile

	virtual FunctionalUnit GetFunctionalUnit(const SASS::Instruction *instruction) const = 0;
	virtual std::uint8_t GetLatency(const SASS::Instruction *instruction) const = 0;
	virtual std::uint8_t GetMinimumLatency(const SASS::Instruction *instruction) const = 0;
	virtual std::uint8_t GetBarrierLatency(const SASS::Instruction *instruction) const = 0;
	virtual std::uint8_t GetReadLatency(const SASS::Instruction *instruction) const = 0;
	virtual std::uint8_t GetReadHold(const SASS::Instruction *instruction) const = 0;
	virtual std::uint8_t GetThroughputLatency(const SASS::Instruction *instruction) const = 0;
	virtual bool GetDualIssue(const SASS::Instruction *instruction) const = 0;
	virtual bool GetReuseFlags(const SASS::Instruction *instruction) const = 0;

	// Barriers

	virtual SASS::Schedule::Barrier GetReadBarrier(const SASS::Instruction *instruction) const = 0;
	virtual SASS::Schedule::Barrier GetWriteBarrier(const SASS::Instruction *instruction) const = 0;
	virtual bool GetScoreboardBarrier(SASS::Schedule::Barrier barrier) const = 0;
};

}
}
