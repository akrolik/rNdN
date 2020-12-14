#pragma once

#include <vector>

#include "HorseIR/Tree/Types/Type.h"

#include "HorseIR/Tree/FunctionDeclaration.h"
#include "HorseIR/Tree/BuiltinFunction.h"
#include "HorseIR/Tree/Function.h"

#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Traversal/ConstHierarchicalVisitor.h"
#include "HorseIR/Traversal/Visitor.h"
#include "HorseIR/Traversal/HierarchicalVisitor.h"

namespace HorseIR {

class FunctionType : public Type
{
public:
	constexpr static Type::Kind TypeKind = Type::Kind::Function;

	enum class FunctionKind {
		Builtin,
		Definition,
		Undefined
	};

	FunctionType() : Type(TypeKind), m_functionKind(FunctionKind::Undefined) {}
	FunctionType(const BuiltinFunction *function) : Type(TypeKind), m_functionKind(FunctionKind::Builtin), m_function(function) {}
	FunctionType(const Function *function) : Type(TypeKind), m_functionKind(FunctionKind::Definition), m_function(function)
	{
		// Assemble parameter types

		for (auto& parameter : function->GetParameters())
		{
			m_parameterTypes.push_back(parameter->GetType());
		}
		m_returnTypes = function->GetReturnTypes();
	}

	FunctionType *Clone() const override
	{
		std::vector<const Type *> returnTypes;
		for (const auto& returnType : m_returnTypes)
		{
			returnTypes.push_back(returnType->Clone());
		}

		std::vector<const Type *> parameterTypes;
		for (const auto& parameterType : parameterTypes)
		{
			parameterTypes.push_back(parameterType->Clone());
		}

		return new FunctionType(m_functionKind, m_function, returnTypes, parameterTypes);
	}

	// Function

	FunctionKind GetFunctionKind() const { return m_functionKind; }
	const FunctionDeclaration *GetFunctionDeclaration() const { return m_function; }

	// Types

	const std::vector<const Type *>& GetReturnTypes() const { return m_returnTypes; }
	const std::vector<const Type *>& GetParameterTypes() const { return m_parameterTypes; }

	// Operators

	bool operator==(const FunctionType& other) const
	{
		if (m_function != nullptr || other.m_function != nullptr)
		{
			return m_function == other.m_function;
		}

		bool ret = std::equal(
			std::begin(m_returnTypes), std::end(m_returnTypes),
			std::begin(other.m_returnTypes), std::end(other.m_returnTypes),
			[](const Type *t1, const Type *t2) { return *t1 == *t2; }
		);
		bool param = std::equal(
			std::begin(m_parameterTypes), std::end(m_parameterTypes),
			std::begin(other.m_parameterTypes), std::end(other.m_parameterTypes),
			[](const Type *t1, const Type *t2) { return *t1 == *t2; }
		);
		return ret && param;
	}

	bool operator!=(const FunctionType& other) const
	{
		return !(*this == other);
	}

	// Visitors

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
	FunctionType(FunctionKind functionKind, const FunctionDeclaration *function, const std::vector<const Type *>& returnTypes, const std::vector<const Type *>& parameterTypes) : Type(TypeKind), m_functionKind(functionKind), m_function(function), m_returnTypes(returnTypes), m_parameterTypes(parameterTypes) {}

	FunctionKind m_functionKind = FunctionKind::Undefined;
	const FunctionDeclaration *m_function = nullptr;

	std::vector<const Type *> m_returnTypes;
	std::vector<const Type *> m_parameterTypes;
};

}
