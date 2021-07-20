#pragma once

#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Scheduler {

class HardwareProperties
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

	static std::uint8_t FunctionalUnitIndex(FunctionalUnit unit)
	{
		return static_cast<std::underlying_type_t<FunctionalUnit>>(unit);
	}

	static FunctionalUnit FunctionalUnitFromIndex(std::uint8_t index)
	{
		return static_cast<FunctionalUnit>(index);
	}

	static FunctionalUnit GetFunctionalUnit(const SASS::Instruction *instruction);
	static std::uint8_t GetLatency(const SASS::Instruction *instruction);
	static std::uint8_t GetMinimumLatency(const SASS::Instruction *instruction);
	static std::uint8_t GetBarrierLatency(const SASS::Instruction *instruction);
	static std::uint8_t GetReadLatency(const SASS::Instruction *instruction);
	static std::uint8_t GetReadHold(const SASS::Instruction *instruction);
	static std::uint8_t GetThroughputLatency(const SASS::Instruction *instruction);
	static bool GetDualIssue(const SASS::Instruction *instruction);
	static bool GetReuseFlags(const SASS::Instruction *instruction);
};

}
}
