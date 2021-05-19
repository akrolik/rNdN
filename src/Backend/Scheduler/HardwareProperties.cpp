#include "Backend/Scheduler/HardwareProperties.h"

namespace Backend {
namespace Scheduler {

std::uint8_t HardwareProperties::GetLatency(const SASS::Instruction *instruction)
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
		case SASS::Instruction::InstructionClass::Integer:
		case SASS::Instruction::InstructionClass::Control:
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

std::uint8_t HardwareProperties::GetMinimumLatency(const SASS::Instruction *instruction)
{
	switch (instruction->GetInstructionClass())
	{
		case SASS::Instruction::InstructionClass::Control:
		{
			return 6; // Must complete
		}
		case SASS::Instruction::InstructionClass::S2R:
		case SASS::Instruction::InstructionClass::GlobalMemoryLoad:
		case SASS::Instruction::InstructionClass::GlobalMemoryStore:
		case SASS::Instruction::InstructionClass::SharedMemoryLoad:
		case SASS::Instruction::InstructionClass::SharedMemoryStore:
		case SASS::Instruction::InstructionClass::DoublePrecision:
		case SASS::Instruction::InstructionClass::SpecialFunction:
		{
			return 2; //TODO: 1 vs 2
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

std::uint8_t HardwareProperties::GetBarrierLatency(const SASS::Instruction *instruction)
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

std::uint8_t HardwareProperties::GetReadLatency(const SASS::Instruction *instruction)
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

std::uint8_t HardwareProperties::GetReadHold(const SASS::Instruction *instruction)
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

std::uint8_t HardwareProperties::GetThroughputLatency(const SASS::Instruction *instruction)
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

bool HardwareProperties::GetDualIssue(const SASS::Instruction *instruction)
{
	//TODO: Dual issue
	return false;
}

bool HardwareProperties::GetReuseFlags(const SASS::Instruction *instruction)
{
	//TODO: Register reuse
	return false;
}

}
}
