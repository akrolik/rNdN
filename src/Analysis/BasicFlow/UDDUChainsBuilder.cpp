#include "Analysis/BasicFlow/UDDUChainsBuilder.h"

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace Analysis {

void UDDUChainsBuilder::Build(const HorseIR::Function *function)
{
	function->Accept(*this);
}

bool UDDUChainsBuilder::VisitIn(const HorseIR::Statement *statement)
{
	m_currentStatement = statement;
	return true;
}

void UDDUChainsBuilder::VisitOut(const HorseIR::Statement *statement)
{
	m_currentStatement = nullptr;
}

bool UDDUChainsBuilder::VisitIn(const HorseIR::AssignStatement *assignS)
{
	// Only traverse the expression

	m_currentStatement = assignS;
	assignS->GetExpression()->Accept(*this);
	return false;
}

bool UDDUChainsBuilder::VisitIn(const HorseIR::FunctionLiteral *literal)
{
	// Skip identifier

	return false;
}

bool UDDUChainsBuilder::VisitIn(const HorseIR::Identifier *identifier)
{
	// Get the in set for the current statement

	const auto& inSet = m_reachingDefinitions.GetInSet(m_currentStatement);

	// By construction of the RD analysis, we already have the UD chain for this identifier

	const auto& definitions = inSet.at(identifier->GetSymbol());
	m_useDefChains[identifier] = *definitions;

	// For each definition, add this as a use if not already present (DU chain)

	for (const auto& definition : *definitions)
	{
		if (m_defUseChains.find(definition) == m_defUseChains.end())
		{
			m_defUseChains[definition] = std::unordered_set<const HorseIR::Identifier *>();
		}
		m_defUseChains[definition].insert(identifier);
	}
	return true;
}

std::string Indent(unsigned int indent)
{
	std::string string;
	for (unsigned int i = 0; i < indent; ++i)
	{
		string += "\t";
	}
	return string;
}

std::string UDDUChainsBuilder::DebugString(unsigned int indent) const
{
	std::string string = Indent(indent) + "UD chains";
	for (const auto& ud : m_useDefChains)
	{
		string += "\n" + Indent(indent + 1);
		string += HorseIR::PrettyPrinter::PrettyString(dynamic_cast<const HorseIR::Operand *>(ud.first));
		string += "->[";

		bool first = true;
		for (const auto& definition : ud.second)
		{
			if (!first)
			{
				string += ", ";
			}
			first = false;
			string += HorseIR::PrettyPrinter::PrettyString(definition, true);
		}
		string += "]";
	}

	string += "\n" + Indent(indent) + "DU chains";
	for (const auto& du : m_defUseChains)
	{
		string += "\n" + Indent(indent + 1);
		string += HorseIR::PrettyPrinter::PrettyString(du.first, true);
		string += "->[";

		bool first = true;
		for (const auto& use : du.second)
		{
			if (!first)
			{
				string += ", ";
			}
			first = false;
			string += HorseIR::PrettyPrinter::PrettyString(dynamic_cast<const HorseIR::Operand *>(use));
		}
		string += "]";
	}
	return string;
}

}
