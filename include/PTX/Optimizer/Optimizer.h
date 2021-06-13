#pragma once

#include "PTX/Traversal/HierarchicalVisitor.h"

#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Optimizer {

class Optimizer : public HierarchicalVisitor
{
public:
	// Public API

	void Optimize(Program *program);

	// Visitors

	bool VisitIn(FunctionDefinition<VoidType> *function) override;
};

}
}
