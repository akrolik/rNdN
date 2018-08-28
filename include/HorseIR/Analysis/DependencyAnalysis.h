#pragma once

#include <string>
#include <unordered_map>

#include "HorseIR/Traversal/ConstForwardTraversal.h"

#include "HorseIR/Analysis/DependencyGraph.h"

namespace HorseIR {

class DependencyAnalysis : public ConstForwardTraversal
{
public:
	void Analyze(const Program *program);
	GlobalDependencyGraph *GetResults() const { return m_graph; }

	void Visit(const Method *method) override;
	void Visit(const Declaration *declaration) override;
	void Visit(const AssignStatement *assign) override;
	void Visit(const ReturnStatement *ret) override;
	void Visit(const Identifier *identifier) override;

private:
	const Statement *m_currentStatement = nullptr;

	GlobalDependencyGraph *m_graph = new GlobalDependencyGraph();
	DependencyGraph *m_dependencies = nullptr;
};

}
