#pragma once

#include "PTX/Traversal/HierarchicalVisitor.h"
#include "PTX/Traversal/ConstDeclarationVisitor.h"

#include "PTX/Analysis/RegisterAllocator/RegisterAllocation.h"
#include "PTX/Tree/Tree.h"

#include "SASS/SASS.h"

namespace Backend {

class Compiler : public PTX::HierarchicalVisitor, public PTX::ConstDeclarationVisitor
{
public:
	SASS::Program *Compile(PTX::Program *program);

	bool VisitIn(PTX::FunctionDefinition<PTX::VoidType> *function) override;

	// Declarations

	bool VisitIn(PTX::VariableDeclaration *declaration) override;

	template<class T, class S>
	void Visit(const PTX::TypedVariableDeclaration<T, S> *declaration);
	void Visit(const PTX::_TypedVariableDeclaration *declaration) override;

private:
	const PTX::Analysis::RegisterAllocation *AllocateRegisters(const PTX::FunctionDefinition<PTX::VoidType> *function);

	SASS::Program *m_program = nullptr;
};

}
