#pragma once

#include <string>
#include <sstream>

#include "PTX/Traversal/ConstHierarchicalVisitor.h"

#include "PTX/Analysis/Framework/FlowAnalysis.h"
#include "PTX/Tree/Tree.h"
#include "PTX/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

namespace PTX {
namespace Analysis {

template<class F, template<class> typename A>
class FlowAnalysisPrinter : public ConstHierarchicalVisitor
{
public:
	static std::string PrettyString(const FlowAnalysis<F, A>& analysis, const FunctionDefinition<VoidType> *function)
	{
		FlowAnalysisPrinter printer(analysis);
		printer.m_string.str("");
		function->Accept(printer);
		return printer.m_string.str();
	}

	bool VisitIn(const Statement *statement) override
	{
		std::string indent = Indent();

		if (m_string.str().length() > 0)
		{
			m_string << std::endl;
		}

		m_string << indent << "------------------------------------------------" << std::endl;
		m_string << indent << PrettyPrinter::PrettyString(statement, true) << std::endl;
		m_string << indent << "------------------------------------------------" << std::endl;

		m_indent++;

		if (m_analysis.CollectInSets())
		{
			m_string << indent << "In: {" << std::endl;
			m_analysis.GetInSet(statement).Print(m_string, m_indent + 1);
			m_string << std::endl;
			m_string << indent << "}";
			
			if (m_analysis.CollectInSets())
			{
				m_string << std::endl;
			}
		}

		if (m_analysis.CollectOutSets())
		{
			m_string << indent << "Out: {" << std::endl;
			m_analysis.GetOutSet(statement).Print(m_string, m_indent + 1);
			m_string << std::endl;
			m_string << indent << "}";
		}

		m_indent--;
		return false;
	}

protected:
	FlowAnalysisPrinter(const FlowAnalysis<F, A>& analysis) : m_analysis(analysis) {}

	const FlowAnalysis<F, A>& m_analysis;

	// String & indentation

	std::stringstream m_string;
	unsigned int m_indent = 0;
	std::string Indent() const
	{
		return std::string(m_indent * Utils::Logger::IndentSize, ' ');
	}
};

}
}
