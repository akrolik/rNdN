#pragma once

#include <queue>
#include <string>
#include <sstream>
#include <unordered_map>

#include "PTX/Tree/Tree.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace PTX {
namespace Analysis {

template<class F>
class ControlFlowAnalysis
{
public:
	void Analyze(const FunctionDefinition<VoidType> *function)
	{
		m_analysisTime = Utils::Chrono::Start(Name() + " '" + function->GetName() + "'");

		auto timeInit_start = Utils::Chrono::Start(Name() + " initial flow");
		auto initialFlow = InitialFlow(function);
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

	// Formatting

	virtual void PrintResults(const FunctionDefinition<VoidType> *function) const;

	// Accessors

	const F& GetInSet(const BasicBlock *block) const { return m_blockInSets.at(block); }
	const F& GetOutSet(const BasicBlock *block) const { return m_blockOutSets.at(block); }

	// Each analysis must provide initial flows and merge operation

	virtual F InitialFlow(const FunctionDefinition<VoidType> *function) const = 0;
	virtual F TemporaryFlow(const FunctionDefinition<VoidType> *function) const = 0;
	virtual F Merge(const F& s1, const F& s2) const = 0;

	virtual std::string Name() const = 0;

protected:
	// Maintain basic block input/output sets

	void SetInSet(const BasicBlock *block, const F& set) { m_blockInSets.insert_or_assign(block, set); }
	void SetOutSet(const BasicBlock *block, const F& set) { m_blockOutSets.insert_or_assign(block, set); }

	bool ContainsInSet(const BasicBlock *block) const { return m_blockInSets.find(block) != m_blockInSets.end(); }
	bool ContainsOutSet(const BasicBlock *block) const { return m_blockOutSets.find(block) != m_blockOutSets.end(); }

	F m_currentInSet;
	F m_currentOutSet;

	std::unordered_map<const BasicBlock *, F> m_blockInSets;
	std::unordered_map<const BasicBlock *, F> m_blockOutSets;

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

#include "PTX/Analysis/Framework/ControlFlowAnalysisPrinter.h"

namespace PTX {
namespace Analysis {

template<class F>
void ControlFlowAnalysis<F>::PrintResults(const FunctionDefinition<VoidType> *function) const
{
	Utils::Logger::LogInfo(Name() + ": " + function->GetName());

	auto string = ControlFlowAnalysisPrinter<F>::PrettyString(*this, function);
	Utils::Logger::LogInfo(string, 0, true, Utils::Logger::NoPrefix);
}

}
}
