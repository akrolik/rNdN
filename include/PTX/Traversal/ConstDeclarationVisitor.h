#pragma once

namespace PTX {

class _TypedVariableDeclaration;
class _InitializedVariableDeclaration;

class ConstDeclarationVisitor
{
public:
	virtual void Visit(const _TypedVariableDeclaration *declaration) {}
	virtual void Visit(const _InitializedVariableDeclaration *declaration) {}
};

}
