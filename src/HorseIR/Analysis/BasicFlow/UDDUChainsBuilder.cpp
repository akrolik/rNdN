#include "HorseIR/Analysis/BasicFlow/UDDUChainsBuilder.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace HorseIR {
namespace Analysis {

void UDDUChainsBuilder::Build(const Function *function)
{
	auto timeUDDU_start = Utils::Chrono::Start("UD/DU chains '" + function->GetName() + "'");
	function->Accept(*this);
	Utils::Chrono::End(timeUDDU_start);

	if (Utils::Options::IsDebug_Print())
	{
		Utils::Logger::LogInfo("UD/DU chains '" + function->GetName() + "'");
		Utils::Logger::LogInfo(DebugString(), 0, true, Utils::Logger::NoPrefix);
	}
}

bool UDDUChainsBuilder::VisitIn(const Statement *statement)
{
	m_currentStatement = statement;
	return true;
}

void UDDUChainsBuilder::VisitOut(const Statement *statement)
{
	m_currentStatement = nullptr;
}

bool UDDUChainsBuilder::VisitIn(const AssignStatement *assignS)
{
	// Only traverse the expression

	m_currentStatement = assignS;
	assignS->GetExpression()->Accept(*this);
	return false;
}

bool UDDUChainsBuilder::VisitIn(const FunctionLiteral *literal)
{
	// Skip identifier

	return false;
}

bool UDDUChainsBuilder::VisitIn(const Identifier *identifier)
{
	// Get the in set for the current statement

	const auto& inSet = m_reachingDefinitions.GetInSet(m_currentStatement);

	// Check if the identifier is a globally defined variable

	if (inSet.find(identifier->GetSymbol()) == inSet.end())
	{
		// Skip global variables

		m_useDefChains[identifier] = std::unordered_set<const AssignStatement *>();
		return true;
	}

	// By construction of the RD analysis, we already have the UD chain for this identifier.

	const auto& definitions = inSet.at(identifier->GetSymbol());
	m_useDefChains[identifier] = *definitions;

	// For each definition, add this as a use if not already present (DU chain)

	for (const auto& definition : *definitions)
	{
		if (m_defUseChains.find(definition) == m_defUseChains.end())
		{
			m_defUseChains[definition] = std::unordered_set<const Identifier *>();
		}
		m_defUseChains[definition].insert(identifier);
	}
	return true;
}

std::string Indent(unsigned int indent)
{
	return std::string(indent * Utils::Logger::IndentSize, ' ');
}

std::string UDDUChainsBuilder::DebugString(unsigned int indent) const
{
	std::string string = Indent(indent) + "UD chains";
	for (const auto& [identifier, definitions] : m_useDefChains)
	{
		string += "\n" + Indent(indent + 1);
		string += PrettyPrinter::PrettyString(identifier);
		string += "->[";

		bool first = true;
		for (const auto& definition : definitions)
		{
			if (!first)
			{
				string += ", ";
			}
			first = false;
			string += PrettyPrinter::PrettyString(definition, true);
		}
		string += "]";
	}

	string += "\n" + Indent(indent) + "DU chains";
	for (const auto& [definition, uses] : m_defUseChains)
	{
		string += "\n" + Indent(indent + 1);
		string += PrettyPrinter::PrettyString(definition, true);
		string += "->[";

		bool first = true;
		for (const auto& use : uses)
		{
			if (!first)
			{
				string += ", ";
			}
			first = false;
			string += PrettyPrinter::PrettyString(use);
		}
		string += "]";
	}
	return string;
}

}
}
