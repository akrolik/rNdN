#include "HorseIR/Analysis/StatementAnalysisPrinter.h"

#include "HorseIR/Utils/PrettyPrinter.h"

namespace HorseIR {

std::string StatementAnalysisPrinter::Indent() const
{
	std::string str;
	for (unsigned int i = 0; i < m_indent; ++i)
	{
		str += "\t";
	}
	return str;
}

std::string StatementAnalysisPrinter::PrettyString(const StatementAnalysis& analysis, const Function *function)
{
	StatementAnalysisPrinter printer(analysis);
	printer.m_string.str("");
	function->Accept(printer);
	return printer.m_string.str();
}

bool StatementAnalysisPrinter::VisitIn(const Statement *statement)
{
	std::string indent = Indent();

	m_string << indent << "------------------------------------------------" << std::endl;
	m_string << indent << PrettyPrinter::PrettyString(statement);
	m_string << indent << "------------------------------------------------" << std::endl;

	m_indent++;
	m_string << m_analysis.DebugString(statement, m_indent) << std::endl;
	m_indent--;
	return false;
}

bool StatementAnalysisPrinter::VisitIn(const IfStatement *ifS)
{
	ConstHierarchicalVisitor::VisitIn(ifS);
	return true;
}

bool StatementAnalysisPrinter::VisitIn(const WhileStatement *whileS)
{
	ConstHierarchicalVisitor::VisitIn(whileS);
	return true;
}

bool StatementAnalysisPrinter::VisitIn(const RepeatStatement *repeatS)
{
	ConstHierarchicalVisitor::VisitIn(repeatS);
	return true;
}

bool StatementAnalysisPrinter::VisitIn(const BlockStatement *blockS)
{
	m_indent++;
	return true;
}

void StatementAnalysisPrinter::VisitOut(const BlockStatement *blockS)
{
	m_indent--;
}

}
