#pragma once

#include "PTX/Traversal/ConstHierarchicalVisitor.h"
#include "PTX/Traversal/ConstDeclarationVisitor.h"

#include "PTX/Analysis/RegisterAllocator/RegisterAllocation.h"
#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Analysis {

class VirtualRegisterAllocator : public ConstHierarchicalVisitor, public ConstDeclarationVisitor
{
public:
	inline const static std::string Name = "Virtual registers allocator";
	inline const static std::string ShortName = "reg";

	// Public API

	void Analyze(const FunctionDefinition<VoidType> *function);
	const RegisterAllocation *GetRegisterAllocation() { return m_allocation; }

	// Functions

	bool VisitIn(const FunctionDefinition<VoidType> *function) override;
	void VisitOut(const FunctionDefinition<VoidType> *function) override;

	// Declarations

	bool VisitIn(const VariableDeclaration *declaration) override;

	template<class T, class S>
	void Visit(const TypedVariableDeclaration<T, S> *declaration);
	void Visit(const _TypedVariableDeclaration *declaration) override;

private:
	std::uint8_t m_registerOffset = 0;
	std::uint8_t m_predicateOffset = 0;
	RegisterAllocation *m_allocation = nullptr;

	const FunctionDefinition<VoidType> *m_currentFunction = nullptr;
};

}
}
