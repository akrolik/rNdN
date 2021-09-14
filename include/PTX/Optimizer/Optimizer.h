#pragma once

#include "PTX/Traversal/HierarchicalVisitor.h"
#include "PTX/Traversal/FunctionVisitor.h"

#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Optimizer {

class Optimizer : public HierarchicalVisitor, public FunctionVisitor
{
public:
	// Public API

	void Optimize(Program *program);
	void Optimize(FunctionDefinition<VoidType> *function);

	// Visitors

	bool VisitIn(Function *function) override;
	void Visit(_FunctionDeclaration *function) override;
	void Visit(_FunctionDefinition *function) override;

	template<class T, class S>
	void Visit(FunctionDefinition<T, S> *function);
};

}
}
