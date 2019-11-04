#include "HorseIR/Semantics/EntryAnalysis.h"

#include "Utils/Logger.h"

namespace HorseIR {

void EntryAnalysis::Analyze(const Program *program)
{
	program->Accept(*this);
}

bool EntryAnalysis::VisitIn(const Function *function)
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
