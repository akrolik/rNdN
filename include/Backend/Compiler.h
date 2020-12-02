#pragma once

#include "PTX/Traversal/ConstHierarchicalVisitor.h"

#include "PTX/Tree/Tree.h"
#include "SASS/SASS.h"

namespace Backend {

class Compiler : public PTX::ConstHierarchicalVisitor
{
public:
	SASS::Program *Compile(const PTX::Program *program);

	bool VisitIn(const PTX::FunctionDefinition<PTX::VoidType> *function) override;

private:
	SASS::Program *m_program = nullptr;
};

}
