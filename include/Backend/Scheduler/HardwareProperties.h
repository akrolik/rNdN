#pragma once

#include "SASS/Tree/Tree.h"

namespace Backend {
namespace Scheduler {

class HardwareProperties
{
public:
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
