#pragma once

#include "PTX/Traversal/ConstHierarchicalVisitor.h"

#include "PTX/Tree/Tree.h"
#include "PTX/Analysis/RegisterAllocator/RegisterAllocation.h"

#include "SASS/SASS.h"

namespace Backend {

class CodeGenerator : public PTX::ConstHierarchicalVisitor
{
public:
	SASS::Program *Generate(const PTX::Program *program, const std::unordered_map<const PTX::FunctionDefinition<PTX::VoidType> *, const PTX::Analysis::RegisterAllocation *> &allocations);

	// Structure

	bool VisitIn(const PTX::Program *program) override;

	// Functions

	bool VisitIn(const PTX::FunctionDefinition<PTX::VoidType> *function) override;
	void VisitOut(const PTX::FunctionDefinition<PTX::VoidType> *function) override;

	// Declarations

	bool VisitIn(const PTX::VariableDeclaration *declaration) override;
	template<class T, class S>
	bool VisitIn(const PTX::TypedVariableDeclaration<T, S> *declaration);

private:
	SASS::Program *m_program = nullptr;
	SASS::Function *m_function = nullptr;

	std::unordered_map<const PTX::FunctionDefinition<PTX::VoidType> *, const PTX::Analysis::RegisterAllocation *> m_allocations;
};

}
