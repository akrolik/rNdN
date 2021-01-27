#pragma once

#include "SASS/SASS.h"

namespace Backend {

class Scheduler
{
public:
	void Schedule(SASS::Function *function)
	{
		for (auto& basicBlock : function->GetBasicBlocks())
		{
			for (auto& instruction : basicBlock->GetInstructions())
			{
				// Both LDG and I2I set barriers

				if (dynamic_cast<SASS::LDGInstruction *>(instruction) || dynamic_cast<SASS::I2IInstruction *>(instruction))
				{
					instruction->SetScheduling(
						15,   // Stall
						true, // Yield
						0,    // Write barrier
						7,    // Read barrier
						0,    // Wait barriers
						0     // Reuse
					);
				}
				else
				{
					instruction->SetScheduling(
						15,   // Stall
						true, // Yield
						7,    // Write barrier
						7,    // Read barrier
						0,    // Wait barriers
						0     // Reuse
					);
				}
			}
		}
	}
};

}
