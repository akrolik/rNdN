#pragma once

#include "Backend/Scheduler/Profiles/HardwareProfile.h"

namespace Backend {
namespace Scheduler {

class Compute61Profile : public HardwareProfile
{
public:
	unsigned int GetComputeCapability() const { return 61; }

	// Profile for compute capability 61 (Pascal 10-series)

	FunctionalUnit GetFunctionalUnit(const SASS::Instruction *instruction) const override;
	std::uint8_t GetLatency(const SASS::Instruction *instruction) const override;
	std::uint8_t GetMinimumLatency(const SASS::Instruction *instruction) const override;
	std::uint8_t GetBarrierLatency(const SASS::Instruction *instruction) const override;
	std::uint8_t GetReadLatency(const SASS::Instruction *instruction) const override;
	std::uint8_t GetReadHold(const SASS::Instruction *instruction) const override;
	std::uint8_t GetThroughputLatency(const SASS::Instruction *instruction) const override;
	bool GetDualIssue(const SASS::Instruction *instruction) const override;
	bool GetReuseFlags(const SASS::Instruction *instruction) const override;
};

}
}
