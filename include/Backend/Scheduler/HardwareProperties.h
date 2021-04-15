#pragma once

#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Scheduler {

class HardwareProperties
{
public:
	static std::uint8_t GetLatency(const SASS::Instruction *instruction);
	static std::uint8_t GetBarrierLatency(const SASS::Instruction *instruction);
	static std::uint8_t GetReadHold(const SASS::Instruction *instruction);

	//TODO: Additional hardware properties
	// std::uint8_t GetReadLatency() const
	// std::uint8_t GetThroughput()
	// bool GetDualIssue()
	// bool GetReuseFlags()
};

}
}
