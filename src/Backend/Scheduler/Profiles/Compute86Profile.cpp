#include "Backend/Scheduler/Profiles/Compute86Profile.h"

namespace Backend {
namespace Scheduler {

Compute86Profile::FunctionalUnit Compute86Profile::GetFunctionalUnit(const SASS::Instruction *instruction) const
{
	switch (instruction->GetInstructionClass())
	{
		case SASS::Instruction::InstructionClass::S2R:
		{
			return FunctionalUnit::S2R;
		}
		case SASS::Instruction::InstructionClass::GlobalMemoryLoad:
		case SASS::Instruction::InstructionClass::GlobalMemoryStore:
		case SASS::Instruction::InstructionClass::SharedMemoryLoad:
		case SASS::Instruction::InstructionClass::SharedMemoryStore:
		{
			return FunctionalUnit::LoadStore;
		}
		case SASS::Instruction::InstructionClass::DoublePrecision:
		{
			return FunctionalUnit::DoublePrecision;
		}
		case SASS::Instruction::InstructionClass::SpecialFunction:
		{
			return FunctionalUnit::SpecialFunction;
		}
		case SASS::Instruction::InstructionClass::Integer:
		case SASS::Instruction::InstructionClass::Shift:
		case SASS::Instruction::InstructionClass::Comparison:
		{
			return FunctionalUnit::I32;
		}
		case SASS::Instruction::InstructionClass::Control:
		case SASS::Instruction::InstructionClass::SinglePrecision:
		{
			return FunctionalUnit::F32;
		}
	}
	return FunctionalUnit::Core;
}

std::uint8_t Compute86Profile::GetLatency(const SASS::Instruction *instruction) const
{
	switch (instruction->GetInstructionClass())
	{
		case SASS::Instruction::InstructionClass::S2R:
		case SASS::Instruction::InstructionClass::GlobalMemoryLoad:
		case SASS::Instruction::InstructionClass::GlobalMemoryStore:
		case SASS::Instruction::InstructionClass::SharedMemoryLoad:
		case SASS::Instruction::InstructionClass::SharedMemoryStore:
		case SASS::Instruction::InstructionClass::DoublePrecision:
		case SASS::Instruction::InstructionClass::SpecialFunction:
		{
			return 2; // 1 cycle issue, 1 cycle barrier
		}
		case SASS::Instruction::InstructionClass::Control:
		{
			return 4; // Fixed pipeline depth
		}
		case SASS::Instruction::InstructionClass::Shift:
		case SASS::Instruction::InstructionClass::Integer:
		case SASS::Instruction::InstructionClass::SinglePrecision:
		{
			return 5; // Fixed pipeline depth
		}
		case SASS::Instruction::InstructionClass::Comparison:
		{
			return 13; // Fixed pipeline depth
		}
	}
	return 15;
}

std::uint8_t Compute86Profile::GetMinimumLatency(const SASS::Instruction *instruction) const
{
	switch (instruction->GetInstructionClass())
	{
		case SASS::Instruction::InstructionClass::Control:
		{
			return 4; // Must complete
		}
		case SASS::Instruction::InstructionClass::S2R:
		case SASS::Instruction::InstructionClass::GlobalMemoryLoad:
		case SASS::Instruction::InstructionClass::GlobalMemoryStore:
		case SASS::Instruction::InstructionClass::SharedMemoryLoad:
		case SASS::Instruction::InstructionClass::SharedMemoryStore:
		case SASS::Instruction::InstructionClass::DoublePrecision:
		case SASS::Instruction::InstructionClass::SpecialFunction:
		{
			return 1; // Immediately available, but must wait 2 cycles before checking barrier
		}
		case SASS::Instruction::InstructionClass::Integer:
		case SASS::Instruction::InstructionClass::Shift:
		case SASS::Instruction::InstructionClass::SinglePrecision:
		case SASS::Instruction::InstructionClass::Comparison:
		{
			return 1; // Immediately available
		}
	}
	return 15;
}

std::uint8_t Compute86Profile::GetBarrierLatency(const SASS::Instruction *instruction) const
{
	switch (instruction->GetInstructionClass())
	{
		case SASS::Instruction::InstructionClass::S2R:
		{
			return 25;
		}
		case SASS::Instruction::InstructionClass::GlobalMemoryLoad:
		{
			return 220;
		}
		case SASS::Instruction::InstructionClass::SharedMemoryLoad:
		{
			return 22;
		}
		case SASS::Instruction::InstructionClass::DoublePrecision:
		{
			return 45;
		}
		case SASS::Instruction::InstructionClass::SpecialFunction:
		{
			return 18;
		}
		case SASS::Instruction::InstructionClass::Control:
		case SASS::Instruction::InstructionClass::Integer:
		case SASS::Instruction::InstructionClass::Shift:
		case SASS::Instruction::InstructionClass::Comparison:
		case SASS::Instruction::InstructionClass::SinglePrecision:
		case SASS::Instruction::InstructionClass::GlobalMemoryStore:
		case SASS::Instruction::InstructionClass::SharedMemoryStore:
		{
			return 0;
		}
	}
	return 0;
}

std::uint8_t Compute86Profile::GetReadLatency(const SASS::Instruction *instruction) const
{
	// All instructions read immediately

	return 0;
}

std::uint8_t Compute86Profile::GetReadHold(const SASS::Instruction *instruction) const
{
	switch (instruction->GetInstructionClass())
	{
		case SASS::Instruction::InstructionClass::S2R:
		case SASS::Instruction::InstructionClass::Control:
		case SASS::Instruction::InstructionClass::Integer:
		case SASS::Instruction::InstructionClass::Comparison:
		case SASS::Instruction::InstructionClass::Shift:
		case SASS::Instruction::InstructionClass::SinglePrecision:
		{
			return 0;
		}
		case SASS::Instruction::InstructionClass::DoublePrecision:
		{
			return 12;
		}
		case SASS::Instruction::InstructionClass::SpecialFunction:
		{
			return 15;
		}
		case SASS::Instruction::InstructionClass::GlobalMemoryLoad:
		{
			return 10;
		}
		case SASS::Instruction::InstructionClass::GlobalMemoryStore:
		{
			return 14;
		}
		case SASS::Instruction::InstructionClass::SharedMemoryLoad:
		{
			return 8;
		}
		case SASS::Instruction::InstructionClass::SharedMemoryStore:
		{
			return 9;
		}
	}
	return 0;
}

std::uint8_t Compute86Profile::GetThroughputLatency(const SASS::Instruction *instruction) const
{
	switch (instruction->GetInstructionClass())
	{
		case SASS::Instruction::InstructionClass::Control:
		case SASS::Instruction::InstructionClass::SinglePrecision:
		{
			return 1;
		}
		case SASS::Instruction::InstructionClass::Integer:
		case SASS::Instruction::InstructionClass::Comparison:
		case SASS::Instruction::InstructionClass::Shift:
		{
			return 2;
		}
		case SASS::Instruction::InstructionClass::DoublePrecision:
		{
			return 64;
		}
		case SASS::Instruction::InstructionClass::S2R:
		case SASS::Instruction::InstructionClass::SpecialFunction:
		case SASS::Instruction::InstructionClass::GlobalMemoryLoad:
		case SASS::Instruction::InstructionClass::GlobalMemoryStore:
		case SASS::Instruction::InstructionClass::SharedMemoryLoad:
		case SASS::Instruction::InstructionClass::SharedMemoryStore:
		{
			return 8;
		}
	}
	return 1;
}

bool Compute86Profile::GetDualIssue(const SASS::Instruction *instruction) const
{
	// Ampere does not support dual issue

	return false;
}

bool Compute86Profile::GetReuseFlags(const SASS::Instruction *instruction) const
{
	switch (instruction->GetInstructionClass())
	{
		case SASS::Instruction::InstructionClass::Integer:
		case SASS::Instruction::InstructionClass::Comparison:
		case SASS::Instruction::InstructionClass::Shift:
		case SASS::Instruction::InstructionClass::SinglePrecision:
		case SASS::Instruction::InstructionClass::DoublePrecision:
		{
			return true;
		}
		case SASS::Instruction::InstructionClass::S2R:
		case SASS::Instruction::InstructionClass::Control:
		case SASS::Instruction::InstructionClass::SpecialFunction:
		case SASS::Instruction::InstructionClass::GlobalMemoryLoad:
		case SASS::Instruction::InstructionClass::GlobalMemoryStore:
		case SASS::Instruction::InstructionClass::SharedMemoryLoad:
		case SASS::Instruction::InstructionClass::SharedMemoryStore:
		{
			return false;
		}
	}
	return false;
}

SASS::Schedule::Barrier Compute86Profile::GetWriteBarrier(const SASS::Instruction *instruction) const
{
	switch (instruction->GetInstructionClass())
	{
		case SASS::Instruction::InstructionClass::GlobalMemoryLoad:
		{
			return SASS::Schedule::Barrier::SB0;
		}
		case SASS::Instruction::InstructionClass::SharedMemoryLoad:
		{
			return SASS::Schedule::Barrier::SB1;
		}
		case SASS::Instruction::InstructionClass::S2R:
		case SASS::Instruction::InstructionClass::DoublePrecision:
		case SASS::Instruction::InstructionClass::SpecialFunction:
		{
			return SASS::Schedule::Barrier::SB2;
		}
	}
	return SASS::Schedule::Barrier::SB0;
}

SASS::Schedule::Barrier Compute86Profile::GetReadBarrier(const SASS::Instruction *instruction) const
{
	switch (instruction->GetInstructionClass())
	{
		case SASS::Instruction::InstructionClass::DoublePrecision:
		case SASS::Instruction::InstructionClass::SpecialFunction:
		{
			return SASS::Schedule::Barrier::SB3;
		}
		case SASS::Instruction::InstructionClass::SharedMemoryLoad:
		case SASS::Instruction::InstructionClass::SharedMemoryStore:
		{
			return SASS::Schedule::Barrier::SB4;
		}
		case SASS::Instruction::InstructionClass::GlobalMemoryLoad:
		case SASS::Instruction::InstructionClass::GlobalMemoryStore:
		{
			return SASS::Schedule::Barrier::SB5;
		}
	}
	return SASS::Schedule::Barrier::SB3;
}

bool Compute86Profile::GetScoreboardBarrier(SASS::Schedule::Barrier barrier) const
{
	// Disabled for Volta onwards

	return false;
}

}
}
