#pragma once

namespace PTX {

class DispatchBase;

class _FunctionDeclaration;
class _FunctionDefinition;

class ConstFunctionVisitor
{
public:
	virtual void Visit(const DispatchBase *function);

	virtual void Visit(const _FunctionDeclaration *declaration);
	virtual void Visit(const _FunctionDefinition *definition);
};

}
