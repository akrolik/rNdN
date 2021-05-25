#include "PTX/Analysis/BasicFlow/LiveIntervals.h"

namespace PTX {
namespace Analysis {

void LiveIntervals::Analyze(const FunctionDefinition<VoidType> *function)
{
	auto& functionName = function->GetName();

	auto timeIntervals_start = Utils::Chrono::Start(Name + " '" + functionName + "'");
	function->Accept(*this);
	Utils::Chrono::End(timeIntervals_start);

	if (Utils::Options::IsBackend_PrintAnalysis(ShortName, functionName))
	{
		Utils::Logger::LogInfo(Name + " '" + functionName + "'");
		Utils::Logger::LogInfo(DebugString(), 0, true, Utils::Logger::NoPrefix);
	}
}

bool LiveIntervals::VisitIn(const InstructionStatement *statement)
{
	// Construct live ranges

	for (const auto& element : m_liveVariables.GetOutSet(statement))
	{
		const auto& name = *element;
		auto it = m_liveIntervals.find(name);
		if (it == m_liveIntervals.end())
		{
			// New live range

			m_liveIntervals.try_emplace(name, m_statementIndex, m_statementIndex);
		}
		else
		{
			// Existing live range, extend

			it->second.second = m_statementIndex;
		}
	}

	// Increment index for intervals

	m_statementIndex++;
	return false;
}

std::string LiveIntervals::DebugString() const
{
	std::vector<std::tuple<std::string, unsigned int, unsigned int>> sortedIntervals;
	for (const auto& [name, interval] : m_liveIntervals)
	{
		sortedIntervals.push_back({ name, interval.first, interval.second });
	}
	
	// Sort live intervals by start position

	std::sort(sortedIntervals.begin(), sortedIntervals.end(), [](auto &left, auto &right)
	{
		return std::get<1>(left) < std::get<1>(right);
	});

	std::string string;
	auto first = true;
	for (const auto& [name, start, end] : sortedIntervals)
	{
		if (!first)
		{
			string += "\n";
		}
		first = false;
		string += "  - " + name + " -> [" + std::to_string(start) + "," + std::to_string(end) + "]";
	}
	return string;
}

}
}
