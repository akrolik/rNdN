#pragma once

#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "HorseIR/Analysis/FlowAnalysis.h"
#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace HorseIR {

template<class F>
class FlowAnalysisPrinter : public ConstHierarchicalVisitor
{
public:
	static std::string PrettyString(const FlowAnalysis<F>& analysis, const Function *function)
	{
		FlowAnalysisPrinter printer(analysis);
		printer.m_string.str("");
		function->Accept(printer);
		return printer.m_string.str();
	}

	FlowAnalysisPrinter(const FlowAnalysis<F>& analysis) : m_analysis(analysis) {}

	std::string Indent() const
	{
		std::string str;
		for (unsigned int i = 0; i < m_indent; ++i)
		{
			str += "\t";
		}
		return str;
	}

	bool VisitIn(const Statement *statement) override
	{
		std::string indent = Indent();

		m_string << indent << "------------------------------------------------" << std::endl;
		m_string << indent << PrettyPrinter::PrettyString(statement);
		m_string << indent << "------------------------------------------------" << std::endl;

		m_indent++;
		indent = Indent();

	        m_string << indent << "In: {" << std::endl;
		m_analysis.GetInSet(statement).Print(m_string, m_indent + 1);
		m_string << std::endl;
		m_string << indent << "}" << std::endl;

	        m_string << indent << "Out: {" << std::endl;
		m_analysis.GetOutSet(statement).Print(m_string, m_indent + 1);
		m_string << std::endl;
		m_string << indent << "}" << std::endl;
		
		m_indent--;
		return false;
	}

	bool VisitIn(const BlockStatement *blockS) override
	{
		m_indent++;
		return true;
	}

	void VisitOut(const BlockStatement *blockS) override
	{
		m_indent--;
	}

	bool VisitIn(const IfStatement *ifS) override
	{
		ConstHierarchicalVisitor::VisitIn(ifS);
		return true;
	}

	bool VisitIn(const WhileStatement *whileS) override
	{
		ConstHierarchicalVisitor::VisitIn(whileS);
		return true;
	}

	bool VisitIn(const RepeatStatement *repeatS) override
	{
		ConstHierarchicalVisitor::VisitIn(repeatS);
		return true;
	}

protected:
	std::stringstream m_string;
	unsigned int m_indent = 0;

	const FlowAnalysis<F>& m_analysis;
};

}
