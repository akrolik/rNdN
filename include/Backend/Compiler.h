#pragma once

#include "PTX/Traversal/ConstHierarchicalVisitor.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

#include "PTX/Analysis/RegisterAllocator/RegisterAllocation.h"

namespace Backend {

class Compiler : public PTX::HierarchicalVisitor
{
public:
	SASS::Program *Compile(PTX::Program *program);

	bool VisitIn(PTX::FunctionDefinition<PTX::VoidType> *function) override;

private:
	const PTX::Analysis::RegisterAllocation *AllocateRegisters(const PTX::FunctionDefinition<PTX::VoidType> *function);

	SASS::Program *m_program = nullptr;
};

}
