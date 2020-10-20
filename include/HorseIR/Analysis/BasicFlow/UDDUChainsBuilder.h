#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "HorseIR/Analysis/BasicFlow/ReachingDefinitions.h"

namespace HorseIR {
namespace Analysis {

class UDDUChainsBuilder : public ConstHierarchicalVisitor
{
public:
	UDDUChainsBuilder(const ReachingDefinitions& reachingDefinitions) : m_reachingDefinitions(reachingDefinitions) {}

	void Build(const Function *function);

	bool VisitIn(const Statement *statement) override;
	void VisitOut(const Statement *statement) override;
	bool VisitIn(const AssignStatement *assignS) override;

	bool VisitIn(const FunctionLiteral *literal) override;
	bool VisitIn(const Identifier *identifier) override;

	const std::unordered_set<const AssignStatement *>& GetDefinitions(const Identifier *identifier) const { return m_useDefChains.at(identifier); }
	const std::unordered_set<const Identifier *>& GetUses(const AssignStatement *assignS) const { return m_defUseChains.at(assignS); }

	std::string DebugString(unsigned int indent = 0) const;

private:
	const ReachingDefinitions& m_reachingDefinitions;
	const Statement *m_currentStatement = nullptr;

	std::unordered_map<const Identifier *, std::unordered_set<const AssignStatement *>> m_useDefChains;
	std::unordered_map<const AssignStatement *, std::unordered_set<const Identifier *>> m_defUseChains;
};

}
}
