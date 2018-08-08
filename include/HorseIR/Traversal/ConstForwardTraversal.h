#pragma once

#include "HorseIR/Traversal/ConstVisitor.h"

namespace HorseIR {

class ConstForwardTraversal : public ConstVisitor
{
public:
	using ConstVisitor::Visit;

	void Visit(const Program *program) override;
	void Visit(const Module *module) override;
	void Visit(const Method *method) override;
	void Visit(const AssignStatement *assign) override;
	void Visit(const Declaration *declaration) override;
	void Visit(const CallExpression *call) override;
	void Visit(const CastExpression *cast) override;
	void Visit(const ReturnStatement *ret) override;
};

}
