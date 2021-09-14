#include "PTX/Traversal/FunctionVisitor.h"

#include "PTX/Tree/Tree.h"

namespace PTX {

void FunctionVisitor::Visit(DispatchBase *function)
{

}

void FunctionVisitor::Visit(_FunctionDeclaration *declaration)
{
	Visit(static_cast<DispatchBase *>(declaration));
}

void FunctionVisitor::Visit(_FunctionDefinition *definition)
{
	Visit(static_cast<DispatchBase *>(definition));
}

}
