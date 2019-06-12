#include "Analysis/EntryAnalysis.h"

#include "Utils/Logger.h"

namespace Analysis {

void EntryAnalysis::Analyze(HorseIR::Program *program)
{
	program->Accept(*this);
}

bool EntryAnalysis::VisitIn(HorseIR::Function *function)
{
	// Check if this is an entry point into the program

	if (function->GetName() == "main")
	{
		// Verify that this is the only entry point

		if (m_entry != nullptr)
		{
			Utils::Logger::LogError("More than one entry point");
		}
		m_entry = function;
	}

	return false;
}

}
