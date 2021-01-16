#include "PTX/Analysis/Framework/BlockAnalysisPrinter.h"

#include "PTX/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace PTX {
namespace Analysis {

std::string BlockAnalysisPrinter::Indent() const
{
	return std::string(m_indent * Utils::Logger::IndentSize, ' ');
}

std::string BlockAnalysisPrinter::PrettyString(const BlockAnalysis& analysis, const FunctionDefinition<VoidType> *function)
{
	BlockAnalysisPrinter printer(analysis);
	printer.m_string.str("");
	function->Accept(printer);
	return printer.m_string.str();
}

bool BlockAnalysisPrinter::VisitIn(const BasicBlock *block)
{
	if (Utils::Options::IsBackend_PrintAnalysisBlock())
	{
		std::string indent = Indent();

		m_string << indent << "------------------------------------------------" << std::endl;
		m_string << indent << PrettyPrinter::PrettyString(block->GetLabel()) << std::endl;
		m_string << indent << "------------------------------------------------" << std::endl;

		m_indent++;
		m_string << m_analysis.DebugString(block, m_indent) << std::endl;
		m_indent--;
		return false;
	}
	return true;
}

bool BlockAnalysisPrinter::VisitIn(const Statement *statement)
{
	std::string indent = Indent();

	m_string << indent << "------------------------------------------------" << std::endl;
	m_string << indent << PrettyPrinter::PrettyString(statement, true) << std::endl;
	m_string << indent << "------------------------------------------------" << std::endl;

	m_indent++;
	m_string << m_analysis.DebugString(statement, m_indent) << std::endl;
	m_indent--;
	return false;
}

}
}
