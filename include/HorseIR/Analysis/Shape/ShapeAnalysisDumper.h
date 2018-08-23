#pragma once

#include <stack>

#include "HorseIR/Traversal/ConstForwardTraversal.h"

#include "HorseIR/Analysis/Shape/Shape.h"
#include "HorseIR/Analysis/Shape/ShapeResults.h"
#include "HorseIR/Tree/Types/Type.h"

namespace HorseIR {

class ShapeAnalysisDumper : public ConstForwardTraversal
{
public:
	void Dump(const MethodDeclaration *method, const ShapeResults *shapes);

	void Visit(const Method *method) override;

	void Visit(const AssignStatement *assign) override;
	void Visit(const ReturnStatement *ret) override;

	void Visit(const Declaration *declaration) override;

	void Visit(const Expression *expression) override;
	void Visit(const CallExpression *call) override;
	void Visit(const CastExpression *cast) override;

private:
	void OpenContext(const CallExpression *call);
	void CloseContext();

	const ShapeResults *m_results = nullptr;
	std::stack<const MethodInvocationShapes *> m_shapes;
	unsigned int m_indentation = 0;
};

}
