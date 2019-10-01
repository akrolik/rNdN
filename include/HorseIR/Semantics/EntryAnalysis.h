#pragma once

#include "HorseIR/Traversal/HierarchicalVisitor.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {

class EntryAnalysis : public HierarchicalVisitor
{
public:
	void Analyze(Program *program);
	Function *GetEntry() const { return m_entry; }

	bool VisitIn(Function *function) override;

private:
	Function *m_entry = nullptr;
};

}
