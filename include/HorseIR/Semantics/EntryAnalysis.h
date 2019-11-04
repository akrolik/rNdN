#pragma once

#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {

class EntryAnalysis : public ConstHierarchicalVisitor
{
public:
	void Analyze(const Program *program);
	const Function *GetEntry() const { return m_entry; }

	bool VisitIn(const Function *function) override;

private:
	const Function *m_entry = nullptr;
};

}
