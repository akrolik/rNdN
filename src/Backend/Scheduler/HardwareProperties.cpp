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
		case SASS::Instruction::HardwareClass::x64:
		{
			return 2;
		}
		case SASS::Instruction::HardwareClass::x32:
		case SASS::Instruction::HardwareClass::Shift:
		{
			return 6;
		}
		case SASS::Instruction::HardwareClass::qtr:
		{
			return 15; //TODO: SG value 8
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
		case SASS::Instruction::HardwareClass::x32:
		case SASS::Instruction::HardwareClass::Shift:
		{
			return 0;
		}
		case SASS::Instruction::HardwareClass::x64:
		{
			return 128;
		}
		case SASS::Instruction::HardwareClass::qtr:
		{
			return 10; //TODO: SG value 0
		}
		case SASS::Instruction::HardwareClass::Compare:
		{
			if (dynamic_cast<const SASS::DSETPInstruction *>(instruction))
			{
				return 10; //TODO: SG  value 0
			}
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
		case SASS::Instruction::HardwareClass::x32:
		case SASS::Instruction::HardwareClass::x64:
		case SASS::Instruction::HardwareClass::qtr:
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
