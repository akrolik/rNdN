#pragma once

#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class ForwardTraversal : public Visitor
{
public:
	using Visitor::Visit;

	void Visit(Program *program) override;
	void Visit(Module *module) override;
	void Visit(Method *method) override;
	void Visit(AssignStatement *assign) override;
	void Visit(Declaration *declaration) override;
	void Visit(CallExpression *call) override;
	void Visit(CastExpression *cast) override;
	void Visit(ReturnStatement *ret) override;
};

}
