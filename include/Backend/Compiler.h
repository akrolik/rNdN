#pragma once

#include "PTX/Traversal/HierarchicalVisitor.h"
#include "PTX/Traversal/ConstDeclarationVisitor.h"
#include "PTX/Traversal/FunctionVisitor.h"

#include "PTX/Analysis/RegisterAllocator/RegisterAllocation.h"
#include "PTX/Tree/Tree.h"

#include "Backend/Scheduler/Profiles/HardwareProfile.h"

#include "SASS/Tree/Tree.h"

namespace Backend {

class Compiler : public PTX::HierarchicalVisitor, public PTX::ConstDeclarationVisitor, public PTX::FunctionVisitor
{
public:
	SASS::Program *Compile(PTX::Program *program);
	SASS::Function *Compile(PTX::FunctionDefinition<PTX::VoidType> *function);

	// Functions

	bool VisitIn(PTX::Function *function) override;
	bool VisitIn(PTX::Module *module) override;

	void Visit(PTX::_FunctionDeclaration *function) override;
	void Visit(PTX::_FunctionDefinition *function) override;

	template<class T, class S>
	void Visit(PTX::FunctionDefinition<T, S> *function);

	// Declarations

	bool VisitIn(PTX::VariableDeclaration *declaration) override;

	template<class T, class S>
	void Visit(const PTX::TypedVariableDeclaration<T, S> *declaration);
	void Visit(const PTX::_TypedVariableDeclaration *declaration) override;

private:
	const PTX::Analysis::RegisterAllocation *AllocateRegisters(const PTX::FunctionDefinition<PTX::VoidType> *function, unsigned int computeCapability);
	const Scheduler::HardwareProfile *GetHardwareProfile(unsigned int computeCapability);

	SASS::Program *m_program = nullptr;
};

}
