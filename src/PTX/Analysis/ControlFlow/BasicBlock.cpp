#include "PTX/Analysis/ControlFlow/BasicBlock.h"

#include "PTX/Utils/PrettyPrinter.h"

namespace PTX {
namespace Analysis {

std::string BasicBlock::ToDOTString() const
{
	if (m_statements.size() == 0)
	{
		return "%empty%";
	}

	if (m_statements.size() <= 3)
	{
		std::string string;
		for (const auto& statement : m_statements)
		{
			string += PrettyPrinter::PrettyString(statement) + "\\l";
		}
		return string;
	}

	std::string string;
	string += PrettyPrinter::PrettyString(m_statements.at(0)) + "\\l";
	string += PrettyPrinter::PrettyString(m_statements.at(1)) + "\\l";
	string += "[...]\\l";
	string += PrettyPrinter::PrettyString(m_statements.back()) + "\\l";
	return string;
}

}
}
