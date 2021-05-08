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
			return 15; //TODO: SG value 13
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
		case SASS::Instruction::HardwareClass::Core:
		case SASS::Instruction::HardwareClass::Control:
		case SASS::Instruction::HardwareClass::Shift:
		{
			return 0;
		}
		case SASS::Instruction::HardwareClass::DoublePrecision:
		{
			return 128;
		}
		case SASS::Instruction::HardwareClass::SpecialFunction:
		{
			return 10; //TODO: SG value 0
		}
		case SASS::Instruction::HardwareClass::Compare:
		{
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
				return 20;
			}
			return 0;
		}
		case SASS::Instruction::HardwareClass::SharedMemory:
		{
			if (dynamic_cast<const SASS::STSInstruction *>(instruction))
			{
				return 20;
			}
			return 0;
		}
	}
	return 0;
}

}
}
