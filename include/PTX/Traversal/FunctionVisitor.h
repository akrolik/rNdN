#pragma once

namespace PTX {

class DispatchBase;

class _FunctionDeclaration;
class _FunctionDefinition;

class FunctionVisitor
{
public:
	virtual void Visit(DispatchBase *function);

	virtual void Visit(_FunctionDeclaration *declaration);
	virtual void Visit(_FunctionDefinition *definition);
};

}
