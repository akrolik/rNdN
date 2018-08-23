#pragma once

#include "HorseIR/Tree/Expressions/Operand.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Tree/MethodDeclaration.h"
#include "HorseIR/Tree/Expressions/ModuleIdentifier.h"
#include "HorseIR/Tree/Types/FunctionType.h"

namespace HorseIR {

class FunctionLiteral : public Operand
{
public:
	FunctionLiteral(ModuleIdentifier *identifier) : Operand(Operand::Kind::Literal), m_identifier(identifier) { SetType(new FunctionType()); }

	ModuleIdentifier *GetIdentifier() const { return m_identifier; }
	void SetIdentifier(ModuleIdentifier *identifier) { m_identifier = identifier; }

	MethodDeclaration *GetMethod() const { return m_method; }
	void SetMethod(MethodDeclaration *method) { m_method = method; }

	std::string ToString() const override
	{
		return "@" + m_identifier->ToString();
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

protected:
	ModuleIdentifier *m_identifier = nullptr;

	MethodDeclaration *m_method = nullptr;
};

}
