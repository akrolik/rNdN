#pragma once

#include <vector>

#include "SASS/Tree/Node.h"
#include "SASS/Tree/Function.h"
#include "SASS/Tree/GlobalVariable.h"
#include "SASS/Tree/SharedVariable.h"
#include "SASS/Tree/DynamicSharedVariable.h"

namespace SASS {

class Program : public Node
{
public:
	// Compute capability

	unsigned int GetComputeCapability() const { return m_computeCapability; }
	void SetComputeCapability(unsigned int computeCapability) { m_computeCapability = computeCapability; }

	// Global memory

	std::vector<const GlobalVariable *> GetGlobalVariables() const
	{
		return { std::begin(m_globalVariables), std::end(m_globalVariables) };
	}
	std::vector<GlobalVariable *>& GetGlobalVariables() { return m_globalVariables; }

	void AddGlobalVariable(GlobalVariable *globalVariable) { m_globalVariables.push_back(globalVariable); }
	void SetGlobalVariables(const std::vector<GlobalVariable *>& globalVariables) { m_globalVariables = globalVariables; }

	// Shared memory

	std::vector<const SharedVariable *> GetSharedVariables() const
	{
		return { std::begin(m_sharedVariables), std::end(m_sharedVariables) };
	}
	std::vector<SharedVariable *>& GetSharedVariables() { return m_sharedVariables; }

	void AddSharedVariable(SharedVariable *sharedVariable) { m_sharedVariables.push_back(sharedVariable); }
	void SetSharedVariables(const std::vector<SharedVariable *>& sharedVariables) { m_sharedVariables = sharedVariables; }

	// Dynamic shared memory

	std::vector<const DynamicSharedVariable *> GetDynamicSharedVariables() const
	{
		return { std::begin(m_dynamicSharedVariables), std::end(m_dynamicSharedVariables) };
	}
	std::vector<DynamicSharedVariable *>& GetDynamicSharedVariables() { return m_dynamicSharedVariables; }

	void AddDynamicSharedVariable(DynamicSharedVariable *sharedVariable) { m_dynamicSharedVariables.push_back(sharedVariable); }
	void SetDynamicSharedVariables(const std::vector<DynamicSharedVariable *>& sharedVariables) { m_dynamicSharedVariables = sharedVariables; }

	// Functions

	std::vector<const Function *> GetFunctions() const
	{
		return { std::begin(m_functions), std::end(m_functions) };
	}
	std::vector<Function *>& GetFunctions() { return m_functions; }

	void AddFunction(Function *function) { m_functions.push_back(function); }
	void SetFunctions(const std::vector<Function *>& functions) { m_functions = functions; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }
	
private:
	unsigned int m_computeCapability = 0;
	std::vector<Function *> m_functions;

	std::vector<GlobalVariable *> m_globalVariables;
	std::vector<SharedVariable *> m_sharedVariables;
	std::vector<DynamicSharedVariable *> m_dynamicSharedVariables;
};

};
