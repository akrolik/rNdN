#pragma once

#include "PTX/Traversal/ConstHierarchicalVisitor.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

namespace Backend {

class Compiler : public PTX::HierarchicalVisitor
{
public:
	SASS::Program *Compile(PTX::Program *program);

	bool VisitIn(PTX::FunctionDefinition<PTX::VoidType> *function) override;

private:
	SASS::Program *m_program = nullptr;
};

}
