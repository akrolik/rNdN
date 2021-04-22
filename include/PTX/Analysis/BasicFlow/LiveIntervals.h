#pragma once

#include "PTX/Traversal/ConstHierarchicalVisitor.h"

#include "PTX/Analysis/BasicFlow/LiveVariables.h"

#include "Libraries/robin_hood.h"

namespace PTX {
namespace Analysis {

class LiveIntervals : public ConstHierarchicalVisitor
{
public:
	using LiveInterval = std::pair<unsigned int, unsigned int>;

	// Public API

	LiveIntervals(const LiveVariables& liveVariables) : m_liveVariables(liveVariables) {}

	void Analyze(const FunctionDefinition<VoidType> *function);

	const robin_hood::unordered_map<std::string, LiveInterval>& GetLiveIntervals() const { return m_liveIntervals; }
	robin_hood::unordered_map<std::string, LiveInterval>& GetLiveIntervals() { return m_liveIntervals; }

	// Visitors

	bool VisitIn(const InstructionStatement *statement) override;

	// Formatting

	std::string DebugString() const;

private:
	const LiveVariables& m_liveVariables;

	unsigned int m_statementIndex = 0;
	robin_hood::unordered_map<std::string, LiveInterval> m_liveIntervals;
};

}
}
