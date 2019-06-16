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

	bool VisitIn(const Statement *statement) override
	{
		m_string << "------------------------------------------------" << std::endl;
		m_string << PrettyPrinter::PrettyString(statement);
		m_string << "------------------------------------------------" << std::endl;

	        m_string << "\tIn: {" << std::endl;
		m_analysis.GetInSet(statement).Print(m_string, 2);
		m_string << std::endl << "\t}" << std::endl;

	        m_string << "\tOut: {" << std::endl;
		m_analysis.GetOutSet(statement).Print(m_string, 2);
		m_string << std::endl << "\t}" << std::endl;
		
		return false;
	}

	bool VisitIn(const BlockStatement *blockS) override
	{
		return true;
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

	const FlowAnalysis<F>& m_analysis;
};

}
