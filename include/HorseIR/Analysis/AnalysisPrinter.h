#pragma once

#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "HorseIR/Analysis/ForwardAnalysis.h"
#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

namespace HorseIR {

template<class F>
class AnalysisPrinter : public ConstHierarchicalVisitor
{
public:
	static std::string PrettyString(const ForwardAnalysis<F>& analysis, const Function *function)
	{
		AnalysisPrinter printer(analysis);
		printer.m_string.str("");
		function->Accept(printer);
		return printer.m_string.str();
	}

	AnalysisPrinter(const ForwardAnalysis<F>& analysis) : m_analysis(analysis) {}

	bool VisitIn(const Statement *statement) override
	{
		m_string << "------------------------------------------------" << std::endl;
		m_string << PrettyPrinter::PrettyString(statement);
		m_string << "------------------------------------------------" << std::endl;

	        m_string << "\tIn: {" << std::endl;
		for (const auto& val : m_analysis.GetInSet(statement))
		{
			m_string << "\t\t" << *val << std::endl;
		}
		m_string << "\t}" << std::endl;

	        m_string << "\tOut: {" << std::endl;
		for (const auto& val : m_analysis.GetOutSet(statement))
		{
			m_string << "\t\t" << *val << std::endl;
		}
		m_string << "\t}" << std::endl << std::endl;
		
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

	const ForwardAnalysis<F>& m_analysis;
};

}
