#pragma once

#include <deque>
#include <string>
#include <sstream>

#include "PTX/Tree/Tree.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

#include "Libraries/robin_hood.h"

namespace PTX {
namespace Analysis {

template<class F>
class ControlFlowAnalysis
{
public:
	ControlFlowAnalysis(const std::string& name, const std::string& shortName) : m_name(name), m_shortName(shortName) {}

	void Analyze(const FunctionDefinition<VoidType> *function)
	{
		m_analysisTime = Utils::Chrono::Start(m_name + " '" + function->GetName() + "'");

		auto timeInit_start = Utils::Chrono::Start(m_name + " initial flow");
		auto initialFlow = InitialFlow(function);
		Utils::Chrono::End(timeInit_start);

		Analyze(function, initialFlow);
	}

	void Analyze(const FunctionDefinition<VoidType> *function, const F& initialFlow)
	{
		// Clear sets and traverse the function, timing results

		auto& functionName = function->GetName();
		if (m_analysisTime == nullptr)
		{
			m_analysisTime = Utils::Chrono::Start(m_name + " '" + functionName + "'");
		}

		m_functionTime = Utils::Chrono::Start(m_name + " '" + functionName + "' body");

		InitializeWorklist(function);

		this->m_currentSet.clear();

		TraverseFunction(function, initialFlow);

		// Print results if needed

		if (Utils::Options::IsBackend_PrintAnalysis(m_shortName, functionName))
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

	// Options

	bool CollectInSets() const { return m_collectInSets; }
	bool CollectOutSets() const { return m_collectOutSets; }

	void SetCollectInSets(bool collectInSets) { m_collectInSets = collectInSets; }
	void SetCollectOutSets(bool collectOutSets) { m_collectOutSets = collectOutSets; }

	const std::string& GetName() const { return m_name; }
	const std::string& GetShortName() const { return m_shortName; }

	// Each analysis must provide initial flows and merge operation

	virtual F InitialFlow(const FunctionDefinition<VoidType> *function) const = 0;
	virtual F TemporaryFlow(const FunctionDefinition<VoidType> *function) const = 0;
	virtual F Merge(const F& s1, const F& s2) const = 0;

protected:
	// Maintain basic block input/output sets

	void SetInSet(const BasicBlock *block, const F& set) { m_blockInSets.insert_or_assign(block, set); }
	void SetOutSet(const BasicBlock *block, const F& set) { m_blockOutSets.insert_or_assign(block, set); }

	void SetInSet(const BasicBlock *block, F&& set) { m_blockInSets.insert_or_assign(block, std::move(set)); }
	void SetOutSet(const BasicBlock *block, F&& set) { m_blockOutSets.insert_or_assign(block, std::move(set)); }

	void SetInSetMove(const BasicBlock *block, F&& set) { m_blockInSets[block] = std::move(set); }
	void SetOutSetMove(const BasicBlock *block, F&& set) { m_blockOutSets[block] = std::move(set); }

	bool ContainsInSet(const BasicBlock *block) const { return m_blockInSets.find(block) != m_blockInSets.end(); }
	bool ContainsOutSet(const BasicBlock *block) const { return m_blockOutSets.find(block) != m_blockOutSets.end(); }

	F m_currentSet;

	robin_hood::unordered_map<const BasicBlock *, F> m_blockInSets;
	robin_hood::unordered_map<const BasicBlock *, F> m_blockOutSets;

	bool m_collectInSets = true;
	bool m_collectOutSets = true;

	// Worklist

	robin_hood::unordered_map<const BasicBlock *, unsigned int> m_blockOrder;

	virtual void InitializeWorklist(const FunctionDefinition<VoidType> *function) = 0;

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
			// https://homepages.dcc.ufmg.br/~fernando/classes/dcc888/ementa/slides/WorkList.pdf

			std::vector<BasicBlock *> sortedWork(std::begin(m_unqueuedWork), std::end(m_unqueuedWork));
			std::sort(std::begin(sortedWork), std::end(sortedWork), [&](auto& left, auto& right)
			{
				return m_blockOrder.at(left) < m_blockOrder.at(right);
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
	robin_hood::unordered_set<BasicBlock *> m_unqueuedWork;

	// Chrono

	const Utils::Chrono::SpanTiming *m_analysisTime = nullptr;
	const Utils::Chrono::SpanTiming *m_functionTime = nullptr;

	// Naming

	const std::string m_name;
	const std::string m_shortName;
};

}
}

#include "PTX/Analysis/Framework/ControlFlowAnalysisPrinter.h"

namespace PTX {
namespace Analysis {

template<class F>
void ControlFlowAnalysis<F>::PrintResults(const FunctionDefinition<VoidType> *function) const
{
	Utils::Logger::LogInfo(m_name + " '" + function->GetName() + "'");

	auto string = ControlFlowAnalysisPrinter<F>::PrettyString(*this, function);
	Utils::Logger::LogInfo(string, 0, true, Utils::Logger::NoPrefix);
}

}
}
