#pragma once

#include "PTX/Tree/Functions/FunctionDeclaration.h"

#include "PTX/Tree/BasicBlock.h"
#include "PTX/Tree/Statements/StatementList.h"
#include "PTX/Tree/Statements/Statement.h"

#include "PTX/Analysis/ControlFlow/ControlFlowGraph.h"
#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraph.h"

#include "Libraries/robin_hood.h"

namespace Frontend { namespace Codegen { class InputOptions; } }

namespace PTX {

DispatchInterface_Space(FunctionDefinition)

template<class RT, class RS = ParameterSpace, bool Assert = true>
class FunctionDefinition : DispatchInherit(FunctionDefinition), public FunctionDeclaration<RT, RS, Assert>, public StatementList
{
public:
	REQUIRE_TYPE_PARAM(FunctionDefinition,
		REQUIRE_BASE(RT, ScalarType) || REQUIRE_EXACT(RT, VoidType)
	);
	REQUIRE_SPACE_PARAM(FunctionDefinition,
		REQUIRE_EXACT(RS, RegisterSpace, ParameterSpace)
	);

	using FunctionDeclaration<RT, RS, Assert>::FunctionDeclaration;

	// Control-flow graph

	const Analysis::ControlFlowGraph *GetControlFlowGraph() const { return m_cfg; }
	Analysis::ControlFlowGraph *GetControlFlowGraph() { return m_cfg; }
	void SetControlFlowGraph(Analysis::ControlFlowGraph *cfg) { m_cfg = cfg; }

	const Analysis::StructureNode *GetStructuredGraph() const { return m_structuredGraph; }
	Analysis::StructureNode *GetStructuredGraph() { return m_structuredGraph; }
	void SetStructuredGraph(Analysis::StructureNode *structuredGraph) { m_structuredGraph = structuredGraph; }

	void InvalidateStatements() { m_statements.clear(); }

	// Basic blocks

	robin_hood::unordered_set<const BasicBlock *> GetBasicBlocks() const
	{
		return { std::begin(m_basicBlocks), std::end(m_basicBlocks) };
	}
	robin_hood::unordered_set<BasicBlock *>& GetBasicBlocks() { return m_basicBlocks; }
	void SetBasicBlocks(const robin_hood::unordered_set<BasicBlock *>& basicBlocks) { m_basicBlocks = basicBlocks; }

	// Options

	unsigned int GetMaxRegisters() const { return m_maxRegisters; }
	void SetMaxRegisters(unsigned int registers) { m_maxRegisters = registers; }

	bool GetThreadsPower2() const { return m_threadsPower2; }
	void SetThreadsPower2(bool threadsPower2) { m_threadsPower2 = threadsPower2; }

	const std::tuple<unsigned int, unsigned int, unsigned int>& GetThreadMultiples() const { return m_threadMultiples; }
	void SetThreadMultiples(unsigned int dimX, unsigned int dimY = 1, unsigned int dimZ = 1) { m_threadMultiples = { dimX, dimY, dimZ }; }

	const std::tuple<unsigned int, unsigned int, unsigned int>& GetRequiredThreads() const { return m_requiredThreads; }
	void SetRequiredThreads(unsigned int dimX, unsigned int dimY = 1, unsigned int dimZ = 1) { m_requiredThreads = { dimX, dimY, dimZ }; }

	const std::tuple<unsigned int, unsigned int, unsigned int>& GetMaxThreads() const { return m_maxThreads; }
	void SetMaxThreads(unsigned int dimX, unsigned int dimY = 1, unsigned int dimZ = 1) { m_maxThreads = { dimX, dimY, dimZ }; }

	unsigned int GetDynamicSharedMemorySize() const { return m_dynamicSharedMemorySize; }
	void SetDynamicSharedMemorySize(unsigned int bytes) { m_dynamicSharedMemorySize = bytes; }

	void SetCodegenOptions(const Frontend::Codegen::InputOptions *codegenOptions) { m_codegenOptions = codegenOptions; }
	const Frontend::Codegen::InputOptions *GetCodegenOptions() const { return m_codegenOptions; }

	// Formatting

	json ToJSON() const override
	{
		json j = FunctionDeclaration<RT, RS, Assert>::ToJSON();
		if (m_cfg == nullptr)
		{
			j["statements"] = StatementList::ToJSON();
		}
		else
		{
			for (const auto& block : m_basicBlocks)
			{
				j["basic_blocks"].push_back(block->ToJSON());
			}
		}
		j["power2_threads"] = m_threadsPower2;
		j["required_threads"] = m_requiredThreads;
		j["max_threads"] = m_maxThreads;
		j["shared_memory"] = m_dynamicSharedMemorySize;
		return j;
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor& visitor) override
	{
		if (visitor.VisitIn(this))
		{
			for (auto& parameter : FunctionDeclaration<RT, RS, Assert>::m_parameters)
			{
				parameter->Accept(visitor);
			}

			if (m_cfg != nullptr)
			{
				m_cfg->LinearOrdering([&](Analysis::ControlFlowNode& block)
				{
					block->Accept(visitor);
				});
			}
			else
			{
				for (auto& statement : m_statements)
				{
					statement->Accept(visitor);
				}
			}
		}
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		if (visitor.VisitIn(this))
		{
			for (const auto& parameter : FunctionDeclaration<RT, RS, Assert>::m_parameters)
			{
				parameter->Accept(visitor);
			}
			if (m_cfg != nullptr)
			{
				m_cfg->LinearOrdering([&](const Analysis::ControlFlowNode& block)
				{
					block->Accept(visitor);
				});
			}
			else
			{
				for (const auto& statement : m_statements)
				{
					statement->Accept(visitor);
				}
			}
		}
		visitor.VisitOut(this);
	}

	void Accept(FunctionVisitor& visitor) override { visitor.Visit(static_cast<_FunctionDefinition *>(this)); }
	void Accept(ConstFunctionVisitor& visitor) const { visitor.Visit(static_cast<const _FunctionDefinition *>(this)); }

protected:
	DispatchMember_Type(RT);
	DispatchMember_Space(RS);

	robin_hood::unordered_set<BasicBlock *> m_basicBlocks;
	Analysis::ControlFlowGraph *m_cfg = nullptr;
	Analysis::StructureNode *m_structuredGraph = nullptr;

	bool m_threadsPower2 = false;
	std::tuple<unsigned int, unsigned int, unsigned int> m_threadMultiples;
	std::tuple<unsigned int, unsigned int, unsigned int> m_requiredThreads;
	std::tuple<unsigned int, unsigned int, unsigned int> m_maxThreads;
	unsigned int m_dynamicSharedMemorySize = 0;
	unsigned int m_maxRegisters = 0;

	const Frontend::Codegen::InputOptions *m_codegenOptions = nullptr;
};

}
