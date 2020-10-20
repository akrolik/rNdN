#pragma once

#include "HorseIR/Traversal/HierarchicalVisitor.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {
namespace Optimizer {

class Optimizer : public HierarchicalVisitor
{
public:
	void Optimize(Program *program);

	bool VisitIn(Function *function) override;

private:
	Program *m_program = nullptr;
};

}
}
