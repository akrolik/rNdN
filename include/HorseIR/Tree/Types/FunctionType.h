#pragma once

#include "HorseIR/Tree/Types/BasicType.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class FunctionType : public BasicType
{
public:
	FunctionType() : BasicType(BasicType::Kind::Function) {}

	MethodDeclaration *GetMethod() const { return m_method; }
	void SetMethod(MethodDeclaration *method) { m_method = method; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

private:
	MethodDeclaration *m_method = nullptr;
};

}
