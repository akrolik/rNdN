#pragma once

#include <queue>
#include <string>
#include <sstream>
#include <unordered_map>

#include "PTX/Analysis/Framework/BlockAnalysis.h"
#include "PTX/Analysis/Framework/BlockAnalysisPrinter.h"
#include "PTX/Traversal/ConstVisitor.h"
#include "PTX/Tree/Tree.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace PTX {
namespace Analysis {

template<class F>
class FlowAnalysis : public ConstVisitor, public BlockAnalysis
{
public:
	void Analyze(const FunctionDefinition<VoidType> *function)
	{
		m_analysisTime = Utils::Chrono::Start(Name() + " '" + function->GetName() + "'");

		auto timeInit_start = Utils::Chrono::Start(Name() + " initial flow");
		auto initialFlow = InitialFlow();
		Utils::Chrono::End(timeInit_start);

		Analyze(function, initialFlow);
	}

	void Analyze(const FunctionDefinition<VoidType> *function, const F& initialFlow)
	{
		// Clear sets and traverse the function, timing results

		if (m_analysisTime == nullptr)
		{
			m_analysisTime = Utils::Chrono::Start(Name() + " '" + function->GetName() + "'");
		}

		m_functionTime = Utils::Chrono::Start(Name() + " '" + function->GetName() + "' body");

		this->m_currentInSet.clear();
		this->m_currentOutSet.clear();

		TraverseFunction(function, initialFlow);

		// Print results if needed

		if (Utils::Options::IsBackend_PrintAnalysis())
		{
			PrintResults(function);
		}

		Utils::Chrono::End(m_functionTime);
		Utils::Chrono::End(m_analysisTime);
	}

	virtual void TraverseFunction(const FunctionDefinition<VoidType> *function, const F& initialFlow) = 0;
	virtual void TraverseBlock(const BasicBlock *block) = 0;
	virtual void TraverseStatements(const std::vector<const Statement *>& statements) = 0;

	virtual void PropagateNext() = 0;

	// Visitors

	virtual void Visit(const Node *node) = 0;
	virtual void Visit(const InstructionStatement *statement) = 0;

	void Visit(const BlockStatement *statement)
	{
		TraverseStatements(statement->GetStatements());
	}

	// Accessors

	const F& GetInSet(const BasicBlock *block) const { return m_blockInSets.at(block); }
	const F& GetOutSet(const BasicBlock *block) const { return m_blockOutSets.at(block); }

	const F& GetInSet(const Statement *statement) const { return m_inSets.at(statement); }
	const F& GetOutSet(const Statement *statement) const { return m_outSets.at(statement); }

	// Formatting

	void PrintResults(const FunctionDefinition<VoidType> *function)
	{
		Utils::Logger::LogInfo(Name() + " " + function->GetName());

		auto string = BlockAnalysisPrinter::PrettyString(*this, function);
		Utils::Logger::LogInfo(string, 0, true, Utils::Logger::NoPrefix);
	}

	std::string DebugString(const Statement *statement, unsigned indent = 0) const
	{
		std::string indentString(indent * Utils::Logger::IndentSize, ' ');

		std::stringstream string;
		string << indentString << "In: {" << std::endl;
		GetInSet(statement).Print(string, indent + 1);
		string << std::endl;
		string << indentString << "}" << std::endl;

		string << indentString << "Out: {" << std::endl;
		GetOutSet(statement).Print(string, indent + 1);
		string << std::endl;
		string << indentString << "}";

		return string.str();
	}

	std::string DebugString(const BasicBlock *block, unsigned indent = 0) const
	{
		std::string indentString(indent * Utils::Logger::IndentSize, ' ');

		std::stringstream string;
		string << indentString << "In: {" << std::endl;
		GetInSet(block).Print(string, indent + 1);
		string << std::endl;
		string << indentString << "}" << std::endl;

		string << indentString << "Out: {" << std::endl;
		GetOutSet(block).Print(string, indent + 1);
		string << std::endl;
		string << indentString << "}";

		return string.str();
	}

	// Each analysis must provide initial flows and merge operation

	virtual F InitialFlow() const = 0;
	virtual F TemporaryFlow() const = 0;
	virtual F Merge(const F& s1, const F& s2) const = 0;

	virtual std::string Name() const = 0;

protected:
	// Maintain both basic block and statement input/output sets

	void SetInSet(const BasicBlock *block, const F& set) { m_blockInSets.insert_or_assign(block, set); }
	void SetOutSet(const BasicBlock *block, const F& set) { m_blockOutSets.insert_or_assign(block, set); }

	void SetInSet(const Statement *statement, const F& set) { m_inSets.insert_or_assign(statement, set); }
	void SetOutSet(const Statement *statement, const F& set) { m_outSets.insert_or_assign(statement, set); }

	bool ContainsInSet(const BasicBlock *block) const { return m_blockInSets.find(block) != m_blockInSets.end(); }
	bool ContainsOutSet(const BasicBlock *block) const { return m_blockOutSets.find(block) != m_blockOutSets.end(); }

	F m_currentInSet;
	F m_currentOutSet;

	std::unordered_map<const BasicBlock *, F> m_blockInSets;
	std::unordered_map<const BasicBlock *, F> m_blockOutSets;

	std::unordered_map<const Statement *, F> m_inSets;
	std::unordered_map<const Statement *, F> m_outSets;

	// Worklist

	bool IsEmptyWorklist() const { return m_worklist.empty(); }
	void PushWorklist(BasicBlock *block) { m_worklist.push(block); }
	BasicBlock *PopWorklist()
	{
		auto element = m_worklist.front();
		m_worklist.pop();
		return element;
	}

	std::queue<BasicBlock *> m_worklist;

	// Chrono

	const Utils::Chrono::SpanTiming *m_analysisTime = nullptr;
	const Utils::Chrono::SpanTiming *m_functionTime = nullptr;
};

}
}
