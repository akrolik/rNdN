#include "PTX/Traversal/ConstFunctionVisitor.h"

#include "PTX/Tree/Tree.h"

namespace PTX {

void ConstFunctionVisitor::Visit(const DispatchBase *function)
{

}

void ConstFunctionVisitor::Visit(const _FunctionDeclaration *declaration)
{
	Visit(static_cast<const DispatchBase *>(declaration));
}

void ConstFunctionVisitor::Visit(const _FunctionDefinition *definition)
{
	Visit(static_cast<const DispatchBase *>(definition));
}

}
