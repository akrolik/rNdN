#include "Backend/Scheduler/Scheduler.h"

namespace Backend {

void Scheduler::Schedule(SASS::Function *function)
{
	for (auto& basicBlock : function->GetBasicBlocks())
	{
		for (auto& instruction : basicBlock->GetInstructions())
		{
			//TODO: 7 should be set to a constant somewhere for no barrier

			auto latency = 15;
			auto barrierLatency = 0;
			auto readHold = 0;

			switch (instruction->GetHardwareClass())
			{
				case SASS::Instruction::HardwareClass::S2R:
				{
					latency = 2;
					barrierLatency = 25;
					break;
				}
				case SASS::Instruction::HardwareClass::GlobalMemory:
				{
					latency = 2;
					if (dynamic_cast<SASS::REDInstruction *>(instruction) || dynamic_cast<SASS::STGInstruction *>(instruction))
					{
						readHold = 20;
					}
					else
					{
						barrierLatency = 200;
					}
					break;
				}
				case SASS::Instruction::HardwareClass::SharedMemory:
				{
					latency = 2;
					if (dynamic_cast<SASS::STSInstruction *>(instruction))
					{
						readHold = 20;
					}
					else
					{
						barrierLatency = 30;
					}
					break;
				}
				case SASS::Instruction::HardwareClass::x32:
				{
					latency = 6;
					barrierLatency = 0;
					break;
				}
				case SASS::Instruction::HardwareClass::x64:
				{
					latency = 2;
					barrierLatency = 128;
					break;
				}
				case SASS::Instruction::HardwareClass::qtr:
				{
					// latency = 8; //TODO: Crashes
					barrierLatency = 10; //TODO: Listed as 0
					break;
				}
				case SASS::Instruction::HardwareClass::Compare:
				{
					if (dynamic_cast<SASS::DSETPInstruction *>(instruction))
					{
						barrierLatency = 10; //TODO: listed as 0
					}
					else
					{
						barrierLatency = 0;
					}
					// latency = 13; //TODO: Crashes
					break;
				}
				case SASS::Instruction::HardwareClass::Shift:
				{
					latency = 6;
					barrierLatency = 0;
					break;
				}
			}
			// latency = 15;

			// virtual Unit GetHardwareUnit() const = 0;
			// virtual std::uint8_t GetHardwareLatency() const = 0;
			// virtual std::uint8_t GetHardwareBarrierLatency() const = 0;
			// virtual std::uint8_t GetHardwareReadLatency() const = 0;
			// virtual std::uint8_t GetHardwareReadHold() const = 0;
			// virtual std::uint8_t GetHardwareThroughput() const = 0;
			// virtual bool GetHardwareDualIssue() const = 0;
			// virtual bool GetHardwareReuseFlags() const = 0;

			if (barrierLatency > 0)
			{
				instruction->SetScheduling(
					latency, // Stall
					true,    // Yield
					0,       // Write barrier
					7,       // Read barrier
					0,       // Wait barriers
					0        // Reuse
				);
			}
			else if (readHold > 0)
			{
				instruction->SetScheduling(
					latency, // Stall
					true,    // Yield
					7,       // Write barrier
					0,       // Read barrier
					0,       // Wait barriers
					0        // Reuse
				);

			}
			else
			{
				instruction->SetScheduling(
					latency, // Stall
					true,    // Yield
					7,       // Write barrier
					7,       // Read barrier
					0,       // Wait barriers
					0        // Reuse
				);
			}
		}
	}
}

}
