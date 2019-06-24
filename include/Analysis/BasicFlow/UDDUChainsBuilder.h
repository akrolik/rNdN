#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "Analysis/BasicFlow/ReachingDefinitions.h"

namespace Analysis {

class UDDUChainsBuilder : public HorseIR::ConstHierarchicalVisitor
{
public:
	UDDUChainsBuilder(const ReachingDefinitions& reachingDefinitions) : m_reachingDefinitions(reachingDefinitions) {}

	void Build(const HorseIR::Function *function);

	bool VisitIn(const HorseIR::Statement *statement) override;
	void VisitOut(const HorseIR::Statement *statement) override;
	bool VisitIn(const HorseIR::AssignStatement *assignS) override;

	bool VisitIn(const HorseIR::FunctionLiteral *literal) override;
	bool VisitIn(const HorseIR::Identifier *identifier) override;

	const std::unordered_set<const HorseIR::AssignStatement *>& GetDefinitions(const HorseIR::Identifier *identifier) { return m_useDefChains.at(identifier); }
	const std::unordered_set<const HorseIR::Identifier *>& GetUses(const HorseIR::AssignStatement *assignS) { return m_defUseChains.at(assignS); }

	std::string DebugString(unsigned int indent = 0) const;

private:
	const ReachingDefinitions& m_reachingDefinitions;
	const HorseIR::Statement *m_currentStatement = nullptr;

	std::unordered_map<const HorseIR::Identifier *, std::unordered_set<const HorseIR::AssignStatement *>> m_useDefChains;
	std::unordered_map<const HorseIR::AssignStatement *, std::unordered_set<const HorseIR::Identifier *>> m_defUseChains;
};

}
