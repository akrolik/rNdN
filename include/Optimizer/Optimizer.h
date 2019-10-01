#pragma once

#include "HorseIR/Traversal/HierarchicalVisitor.h"

#include "HorseIR/Tree/Tree.h"

namespace Optimizer {

class Optimizer : public HorseIR::HierarchicalVisitor
{
public:
	void Optimize(HorseIR::Program *program);

	bool VisitIn(HorseIR::Function *function) override;

private:
	HorseIR::Program *m_program = nullptr;
};

}
