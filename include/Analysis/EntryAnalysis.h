#pragma once

#include "HorseIR/Traversal/HierarchicalVisitor.h"

#include "HorseIR/Tree/Tree.h"

namespace Analysis {

class EntryAnalysis : public HorseIR::HierarchicalVisitor
{
public:
	void Analyze(HorseIR::Program *program);
	HorseIR::Function *GetEntry() const { return m_entry; }

	bool VisitIn(HorseIR::Function *function) override;

private:
	HorseIR::Function *m_entry = nullptr;
};

}
