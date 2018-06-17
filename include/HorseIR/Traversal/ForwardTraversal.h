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
	void Visit(CallExpression *call) override;
};

}
