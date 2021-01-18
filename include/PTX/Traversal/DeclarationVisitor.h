#pragma once

namespace PTX {

class _TypedVariableDeclaration;
class _InitializedVariableDeclaration;

class DeclarationVisitor
{
public:
	virtual void Visit(_TypedVariableDeclaration *declaration) {}
	virtual void Visit(_InitializedVariableDeclaration *declaration) {}
};

}
