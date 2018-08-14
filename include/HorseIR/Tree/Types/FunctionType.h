#pragma once

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/Visitor.h"

namespace HorseIR {

class FunctionType : public Type
{
public:
	constexpr static Type::Kind TypeKind = Type::Kind::Function;

	FunctionType() : Type(TypeKind) {}

	std::string ToString() const override
	{
		return "func";
	}

	MethodDeclaration *GetMethod() const { return m_method; }
	void SetMethod(MethodDeclaration *method) { m_method = method; }

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	bool operator==(const FunctionType& other) const
	{
		return true;
	}

	bool operator!=(const FunctionType& other) const
	{
		return false;
	}

private:
	MethodDeclaration *m_method = nullptr;
};

}
