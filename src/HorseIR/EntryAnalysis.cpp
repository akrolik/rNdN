#include "HorseIR/EntryAnalysis.h"

#include "Utils/Logger.h"

namespace HorseIR {

void EntryAnalysis::Analyze(Program *program)
{
	program->Accept(*this);
}

void EntryAnalysis::Visit(Method *method)
{
	// Check if this is an entry point into the program

	if (method->GetName() != "main")
	{
		return;
	}

	// Verify that this is the only entry point

	if (m_entry != nullptr)
	{
		Utils::Logger::LogError("More than one entry point");
	}
	m_entry = method;
}

}
