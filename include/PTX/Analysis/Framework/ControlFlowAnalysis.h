#pragma once

#include <deque>
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

		InitializeWorklist(function);

		this->m_currentSet.clear();

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

	F m_currentSet;

	std::unordered_map<const BasicBlock *, F> m_blockInSets;
	std::unordered_map<const BasicBlock *, F> m_blockOutSets;

	// Worklist

	std::unordered_map<const BasicBlock *, unsigned int> m_blockOrder;

	void InitializeWorklist(const FunctionDefinition<VoidType> *function)
	{
		const auto cfg = function->GetControlFlowGraph();
		const auto entry = cfg->GetEntryNode();
		auto index = cfg->GetNodeCount();

		cfg->DFS(entry, [&](BasicBlock *block)
		{
			m_blockOrder[block] = index;
			index--;

			return false;
		}, Utils::Graph<BasicBlock *>::Traversal::Postorder);
	}

	bool IsEmptyWorklist() const
	{
		return m_queuedWork.empty() && m_unqueuedWork.empty();
	}
	void PushWorklist(BasicBlock *block)
	{
		if (std::find(m_queuedWork.begin(), m_queuedWork.end(), block) == m_queuedWork.end())
		{
			m_unqueuedWork.insert(block);
		}
	}
	BasicBlock *PopWorklist()
	{
		if (m_queuedWork.empty())
		{
			// Sort the unqueued work by reverse post-order

			std::vector<BasicBlock *> sortedWork(std::begin(m_unqueuedWork), std::end(m_unqueuedWork));
			std::sort(std::begin(sortedWork), std::end(sortedWork), [&](auto& left, auto& right)
			{
				return m_blockOrder[left] < m_blockOrder[right];
			});

			// Queue the sorted work

			for (auto& block : sortedWork)
			{
				m_queuedWork.push_back(block);
			}

			m_unqueuedWork.clear();
		}

		// Get the next queued element

		auto element = m_queuedWork.front();
		m_queuedWork.pop_front();
		return element;
	}

	std::deque<BasicBlock *> m_queuedWork;
	std::unordered_set<BasicBlock *> m_unqueuedWork;

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
