#include "Backend/Scheduler/HardwareProperties.h"

namespace Backend {
namespace Scheduler {

std::uint8_t HardwareProperties::GetLatency(const SASS::Instruction *instruction)
{
	switch (instruction->GetHardwareClass())
	{
		case SASS::Instruction::HardwareClass::S2R:
		case SASS::Instruction::HardwareClass::GlobalMemory:
		case SASS::Instruction::HardwareClass::SharedMemory:
		case SASS::Instruction::HardwareClass::DoublePrecision:
		case SASS::Instruction::HardwareClass::SpecialFunction:
		{
			return 2; // 1 cycle + barrier
		}
		case SASS::Instruction::HardwareClass::Core:
		case SASS::Instruction::HardwareClass::Control:
		case SASS::Instruction::HardwareClass::Shift:
		{
			return 6; // Fixed pipeline depth
		}
		case SASS::Instruction::HardwareClass::Compare:
		{
			return 13; // Fixed pipeline depth
		}
	}
	return 15;
}

std::uint8_t HardwareProperties::GetMinimumLatency(const SASS::Instruction *instruction)
{
	switch (instruction->GetHardwareClass())
	{
		case SASS::Instruction::HardwareClass::S2R:
		case SASS::Instruction::HardwareClass::GlobalMemory:
		case SASS::Instruction::HardwareClass::SharedMemory:
		case SASS::Instruction::HardwareClass::DoublePrecision:
		case SASS::Instruction::HardwareClass::SpecialFunction:
		{
			return 2; // 1 cycle + barrier
		}
		case SASS::Instruction::HardwareClass::Control:
		{
			return 6; // Must complete
		}
		case SASS::Instruction::HardwareClass::Core:
		case SASS::Instruction::HardwareClass::Shift:
		case SASS::Instruction::HardwareClass::Compare:
		{
			if (dynamic_cast<const SASS::DEPBARInstruction *>(instruction))
			{
				return 6;
			}
			return 1; // Immediately available
		}
	}
	return 15;
}

std::uint8_t HardwareProperties::GetBarrierLatency(const SASS::Instruction *instruction)
{
	switch (instruction->GetHardwareClass())
	{
		case SASS::Instruction::HardwareClass::S2R:
		{
			return 25;
		}
		case SASS::Instruction::HardwareClass::GlobalMemory:
		{
			if (dynamic_cast<const SASS::REDInstruction *>(instruction) || dynamic_cast<const SASS::STGInstruction *>(instruction))
			{
				return 0;
			}
			return 200;
		}
		case SASS::Instruction::HardwareClass::SharedMemory:
		{
			if (dynamic_cast<const SASS::STSInstruction *>(instruction))
			{
				return 0;
			}
			return 30;
		}
		case SASS::Instruction::HardwareClass::DoublePrecision:
		{
			return 32;
		}
		case SASS::Instruction::HardwareClass::SpecialFunction:
		{
			return 16; //TODO: SG lists value as 0
		}
		case SASS::Instruction::HardwareClass::Core:
		case SASS::Instruction::HardwareClass::Control:
		case SASS::Instruction::HardwareClass::Shift:
		case SASS::Instruction::HardwareClass::Compare:
		{
			return 0;
		}
	}
	return 0;
}

std::uint8_t HardwareProperties::GetReadLatency(const SASS::Instruction *instruction)
{
	switch (instruction->GetHardwareClass())
	{
		case SASS::Instruction::HardwareClass::S2R:
		case SASS::Instruction::HardwareClass::Control:
		case SASS::Instruction::HardwareClass::Core:
		case SASS::Instruction::HardwareClass::DoublePrecision:
		case SASS::Instruction::HardwareClass::SpecialFunction:
		case SASS::Instruction::HardwareClass::Compare:
		case SASS::Instruction::HardwareClass::Shift:
		{
			return 0;
		}
		case SASS::Instruction::HardwareClass::GlobalMemory:
		{
			//TODO: Read latency RED/ATOM
			if (dynamic_cast<const SASS::STGInstruction *>(instruction))
			{
				// return 4;
			}
			return 0;
		}
		case SASS::Instruction::HardwareClass::SharedMemory:
		{
			//TODO: Read latency STS
			// if (dynamic_cast<const SASS::STSInstruction *>(instruction))
			// {
			// 	return 2;
			// }
			return 0;
		}
	}
	return 0;
}

std::uint8_t HardwareProperties::GetReadHold(const SASS::Instruction *instruction)
{
	switch (instruction->GetHardwareClass())
	{
		case SASS::Instruction::HardwareClass::S2R:
		case SASS::Instruction::HardwareClass::Control:
		case SASS::Instruction::HardwareClass::Core:
		case SASS::Instruction::HardwareClass::DoublePrecision:
		case SASS::Instruction::HardwareClass::SpecialFunction:
		case SASS::Instruction::HardwareClass::Compare:
		case SASS::Instruction::HardwareClass::Shift:
		{
			return 0;
		}
		case SASS::Instruction::HardwareClass::GlobalMemory:
		{
			if (dynamic_cast<const SASS::REDInstruction *>(instruction) || dynamic_cast<const SASS::STGInstruction *>(instruction))
			{
				return 15;
			}
			return 0;
		}
		case SASS::Instruction::HardwareClass::SharedMemory:
		{
			if (dynamic_cast<const SASS::STSInstruction *>(instruction))
			{
				return 15;
			}
			return 0;
		}
	}
	return 0;
}

std::uint8_t HardwareProperties::GetThroughputLatency(const SASS::Instruction *instruction)
{
	switch (instruction->GetHardwareClass())
	{
		case SASS::Instruction::HardwareClass::S2R:
		case SASS::Instruction::HardwareClass::Control:
		case SASS::Instruction::HardwareClass::Core:
		{
			return 1;
		}
		case SASS::Instruction::HardwareClass::Compare:
		case SASS::Instruction::HardwareClass::Shift:
		{
			return 2;
		}
		case SASS::Instruction::HardwareClass::DoublePrecision:
		{
			return 16;
		}
		case SASS::Instruction::HardwareClass::SpecialFunction:
		case SASS::Instruction::HardwareClass::GlobalMemory:
		case SASS::Instruction::HardwareClass::SharedMemory:
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
