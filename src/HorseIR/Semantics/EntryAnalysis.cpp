#include "HorseIR/Semantics/EntryAnalysis.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace HorseIR {

void EntryAnalysis::Analyze(const Program *program)
{
	auto timeEntry_start = Utils::Chrono::Start("Entry analysis");

	program->Accept(*this);

	if (m_entry == nullptr)
	{
		Utils::Logger::LogError("Entry point not found");
	}

	Utils::Chrono::End(timeEntry_start);
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
