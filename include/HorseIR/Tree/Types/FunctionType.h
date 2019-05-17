#pragma once

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class FunctionType : public Type
{
public:
	constexpr static Type::Kind TypeKind = Type::Kind::Function;

	FunctionType() : Type(TypeKind) {}

	FunctionDeclaration *GetFunction() const { return m_function; }
	void SetFunction(FunctionDeclaration *function) { m_function = function; }

	bool operator==(const FunctionType& other) const
	{
		return (m_function == other.m_function);
	}

	bool operator!=(const FunctionType& other) const
	{
		return !(*this == other);
	}

	void Accept(Visitor &visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor &visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor &visitor) override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

	void Accept(ConstHierarchicalVisitor &visitor) const override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

protected:
	FunctionDeclaration *m_function = nullptr;
};

}
