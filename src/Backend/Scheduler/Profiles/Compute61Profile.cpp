#include "Backend/Scheduler/Profiles/Compute61Profile.h"

namespace Backend {
namespace Scheduler {

Compute61Profile::FunctionalUnit Compute61Profile::GetFunctionalUnit(const SASS::Instruction *instruction) const
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
		case SASS::Instruction::InstructionClass::SinglePrecision:
		case SASS::Instruction::InstructionClass::Control:
		{
			return FunctionalUnit::Core;
		}
		case SASS::Instruction::InstructionClass::Shift:
		case SASS::Instruction::InstructionClass::Comparison:
		{
			return FunctionalUnit::HalfCore;
		}
	}
	return FunctionalUnit::Core;
}

std::uint8_t Compute61Profile::GetLatency(const SASS::Instruction *instruction) const
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
			return 5;
		}
		case SASS::Instruction::InstructionClass::Integer:
		case SASS::Instruction::InstructionClass::Shift:
		{
			return 6; // Fixed pipeline depth
		}
		case SASS::Instruction::InstructionClass::Comparison:
		{
			return 13; // Fixed pipeline depth
		}
	}
	return 15;
}

std::uint8_t Compute61Profile::GetMinimumLatency(const SASS::Instruction *instruction) const
{
	switch (instruction->GetInstructionClass())
	{
		case SASS::Instruction::InstructionClass::Control:
		{
			return 5; // Must complete
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
		case SASS::Instruction::InstructionClass::Comparison:
		{
			return 1; // Immediately available
		}
	}
	return 15;
}

std::uint8_t Compute61Profile::GetBarrierLatency(const SASS::Instruction *instruction) const
{
	switch (instruction->GetInstructionClass())
	{
		case SASS::Instruction::InstructionClass::S2R:
		{
			return 25;
		}
		case SASS::Instruction::InstructionClass::GlobalMemoryLoad:
		{
			return 200;
		}
		case SASS::Instruction::InstructionClass::SharedMemoryLoad:
		{
			return 30;
		}
		case SASS::Instruction::InstructionClass::DoublePrecision:
		{
			return 32;
		}
		case SASS::Instruction::InstructionClass::SpecialFunction:
		{
			return 16;
		}
		case SASS::Instruction::InstructionClass::Integer:
		case SASS::Instruction::InstructionClass::Control:
		case SASS::Instruction::InstructionClass::Shift:
		case SASS::Instruction::InstructionClass::Comparison:
		case SASS::Instruction::InstructionClass::GlobalMemoryStore:
		case SASS::Instruction::InstructionClass::SharedMemoryStore:
		{
			return 0;
		}
	}
	return 0;
}

std::uint8_t Compute61Profile::GetReadLatency(const SASS::Instruction *instruction) const
{
	switch (instruction->GetInstructionClass())
	{
		case SASS::Instruction::InstructionClass::S2R:
		case SASS::Instruction::InstructionClass::Control:
		case SASS::Instruction::InstructionClass::Integer:
		case SASS::Instruction::InstructionClass::DoublePrecision:
		case SASS::Instruction::InstructionClass::SpecialFunction:
		case SASS::Instruction::InstructionClass::Comparison:
		case SASS::Instruction::InstructionClass::Shift:
		{
			return 0;
		}
		case SASS::Instruction::InstructionClass::GlobalMemoryLoad:
		case SASS::Instruction::InstructionClass::GlobalMemoryStore:
		{
			return 4;
		}
		case SASS::Instruction::InstructionClass::SharedMemoryLoad:
		case SASS::Instruction::InstructionClass::SharedMemoryStore:
		{
			return 2;
		}
	}
	return 0;
}

std::uint8_t Compute61Profile::GetReadHold(const SASS::Instruction *instruction) const
{
	switch (instruction->GetInstructionClass())
	{
		case SASS::Instruction::InstructionClass::S2R:
		case SASS::Instruction::InstructionClass::Control:
		case SASS::Instruction::InstructionClass::Integer:
		case SASS::Instruction::InstructionClass::Comparison:
		case SASS::Instruction::InstructionClass::Shift:
		{
			return 0;
		}
		case SASS::Instruction::InstructionClass::DoublePrecision:
		case SASS::Instruction::InstructionClass::SpecialFunction:
		case SASS::Instruction::InstructionClass::GlobalMemoryLoad:
		case SASS::Instruction::InstructionClass::GlobalMemoryStore:
		case SASS::Instruction::InstructionClass::SharedMemoryLoad:
		case SASS::Instruction::InstructionClass::SharedMemoryStore:
		{
			return 15; // Unbuffered reads
		}
	}
	return 0;
}

std::uint8_t Compute61Profile::GetThroughputLatency(const SASS::Instruction *instruction) const
{
	switch (instruction->GetInstructionClass())
	{
		case SASS::Instruction::InstructionClass::S2R:
		case SASS::Instruction::InstructionClass::Control:
		case SASS::Instruction::InstructionClass::Integer:
		{
			return 1;
		}
		case SASS::Instruction::InstructionClass::Comparison:
		case SASS::Instruction::InstructionClass::Shift:
		{
			return 2;
		}
		case SASS::Instruction::InstructionClass::DoublePrecision:
		{
			return 16;
		}
		case SASS::Instruction::InstructionClass::SpecialFunction:
		case SASS::Instruction::InstructionClass::GlobalMemoryLoad:
		case SASS::Instruction::InstructionClass::GlobalMemoryStore:
		case SASS::Instruction::InstructionClass::SharedMemoryLoad:
		case SASS::Instruction::InstructionClass::SharedMemoryStore:
		{
			return 4;
		}
	}
	return 1;
}

bool Compute61Profile::GetDualIssue(const SASS::Instruction *instruction) const
{
	switch (instruction->GetInstructionClass())
	{
		case SASS::Instruction::InstructionClass::S2R:
		case SASS::Instruction::InstructionClass::Control:
		case SASS::Instruction::InstructionClass::Integer:
		case SASS::Instruction::InstructionClass::Comparison:
		case SASS::Instruction::InstructionClass::Shift:
		case SASS::Instruction::InstructionClass::DoublePrecision:
		{
			return false;
		}
		case SASS::Instruction::InstructionClass::SpecialFunction:
		case SASS::Instruction::InstructionClass::GlobalMemoryLoad:
		case SASS::Instruction::InstructionClass::GlobalMemoryStore:
		case SASS::Instruction::InstructionClass::SharedMemoryLoad:
		case SASS::Instruction::InstructionClass::SharedMemoryStore:
		{
			return true;
		}
	}
	return false;
}

bool Compute61Profile::GetReuseFlags(const SASS::Instruction *instruction) const
{
	switch (instruction->GetInstructionClass())
	{
		case SASS::Instruction::InstructionClass::Integer:
		case SASS::Instruction::InstructionClass::Comparison:
		case SASS::Instruction::InstructionClass::Shift:
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

}
}
