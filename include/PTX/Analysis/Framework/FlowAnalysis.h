#pragma once

#include <string>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "PTX/Traversal/ConstVisitor.h"
#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Analysis {

template<class F, template<class> typename A>
class FlowAnalysis : public A<F>, public ConstVisitor
{
public:
	// Analysis framework

	void TraverseBlock(const BasicBlock *block) override
	{
		TraverseStatements(block->GetStatements());
	}

	virtual void TraverseStatements(const std::vector<const Statement *>& statements) = 0;

	// Visitors

	virtual void Visit(const Node *node) = 0;

	void Visit(const BlockStatement *statement) override
	{
		TraverseStatements(statement->GetStatements());
	}

	void Visit(const DeclarationStatement *statement) override
	{
		statement->GetDeclaration()->Accept(*this);
	}

	void Visit(const InstructionStatement *statement) override
	{
		for (const auto& operand : statement->GetOperands())
		{
			operand->Accept(*this);
		}
	}

	void Visit(const LabelStatement *statement) override
	{
		statement->GetLabel()->Accept(*this);
	}

	// Accessors

	const F& GetInSet(const Statement *statement) const { return m_inSets.at(statement); }
	const F& GetOutSet(const Statement *statement) const { return m_outSets.at(statement); }

	// Formatting

	void PrintResults(const FunctionDefinition<VoidType> *function) const override;

	using A<F>::Name;

protected:
	// Maintain statement input/output sets

	void SetInSet(const Statement *statement, const F& set) { m_inSets.insert_or_assign(statement, set); }
	void SetOutSet(const Statement *statement, const F& set) { m_outSets.insert_or_assign(statement, set); }

	std::unordered_map<const Statement *, F> m_inSets;
	std::unordered_map<const Statement *, F> m_outSets;
};

}
}

#include "PTX/Analysis/Framework/FlowAnalysisPrinter.h"

namespace PTX {
namespace Analysis {

template<class F, template<class> typename A>
void FlowAnalysis<F, A>::PrintResults(const FunctionDefinition<VoidType> *function) const
{
	if (Utils::Options::IsBackend_PrintAnalysisBlock())
	{
		A<F>::PrintResults(function);
	}
	else
	{
		Utils::Logger::LogInfo(Name() + ": " + function->GetName());

		auto string = FlowAnalysisPrinter<F, A>::PrettyString(*this, function);
		Utils::Logger::LogInfo(string, 0, true, Utils::Logger::NoPrefix);
	}
}

}
}
