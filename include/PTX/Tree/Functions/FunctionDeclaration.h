#pragma once

#include <tuple>
#include <sstream>

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Tuple.h"
#include "PTX/Tree/Type.h"
#include "PTX/Tree/Declarations/Declaration.h"
#include "PTX/Tree/Declarations/VariableDeclaration.h"
#include "PTX/Tree/Functions/FunctionDeclarationBase.h"

namespace PTX {

class VoidType;

template<class R>
class FunctionDeclaration : public FunctionDeclarationBase<R>
{
public:
	using FunctionDeclarationBase<R>::FunctionDeclarationBase;
	using Signature = R;

	// Properties

	std::vector<const VariableDeclaration *> GetParameters() const override
	{
		return { std::begin(m_parameters), std::end(m_parameters) };
	}
	std::vector<VariableDeclaration *>& GetParameters() override { return m_parameters; }

	template<class T, class S>
	std::enable_if_t<REQUIRE_EXACT(S, RegisterSpace) || REQUIRE_BASE(S, ParameterSpace), void>
	AddParameter(TypedVariableDeclaration<T, S> *parameter) { m_parameters.push_back(parameter); }

	// Formatting

	json ToJSON() const override
	{
		json j = FunctionDeclarationBase<R>::ToJSON();
		for (const auto& parameter : m_parameters)
		{
			j["parameters"].push_back(parameter->ToJSON());
		}
		return j;
	}

	// Visitors

	void Accept(Visitor& visitor) override
	{
		if constexpr(std::is_same<R, VoidType>::value)
		{
			visitor.Visit(this);
		}
	}

	void Accept(ConstVisitor& visitor) const override
	{
		if constexpr(std::is_same<R, VoidType>::value)
		{
			visitor.Visit(this);
		}
	}

	void Accept(HierarchicalVisitor& visitor) override
	{
		if constexpr(std::is_same<R, VoidType>::value)
		{
			if (visitor.VisitIn(this))
			{
				for (auto& parameter : m_parameters)
				{
					parameter->Accept(visitor);
				}
			}
			visitor.VisitOut(this);
		}
	}

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		if constexpr(std::is_same<R, VoidType>::value)
		{
			if (visitor.VisitIn(this))
			{
				for (const auto& parameter : m_parameters)
				{
					parameter->Accept(visitor);
				}
			}
			visitor.VisitOut(this);
		}
	}

protected:
	std::vector<VariableDeclaration *> m_parameters;
};

template<class R, typename... Args>
class FunctionDeclaration<R(Args...)> : public FunctionDeclarationBase<R>
{
public:
	REQUIRE_SPACE_PARAM(FunctionDeclaration,
		is_all<REQUIRE_EXACT(typename Args::VariableSpace, RegisterSpace) || REQUIRE_BASE(typename Args::VariableSpace, ParameterSpace)...>::value
	);

	using Signature = R(Args...);

	FunctionDeclaration() {}
	FunctionDeclaration(const std::string& name, typename FunctionDeclarationBase<R>::ReturnDeclarationType *ret, TypedVariableDeclaration<typename Args::VariableType, typename Args::VariableSpace>* ...parameters, Declaration::LinkDirective linkDirective = Declaration::LinkDirective::None) : FunctionDeclarationBase<R>(name, ret, linkDirective)
	{
		auto tuple = std::make_tuple(parameters...);
		ExpandTuple(m_parameters, tuple, int_<sizeof...(Args)>());
	}

	// Properties

	std::vector<const VariableDeclaration *> GetParameters() const override
	{
		return { std::begin(m_parameters), std::end(m_parameters) };
	}
	std::vector<VariableDeclaration *>& GetParameters() override { return m_parameters; }

	void SetParameters(TypedVariableDeclaration<typename Args::VariableType, typename Args::VariableSpace>* ...parameters)
	{
		auto tuple = std::make_tuple(parameters...);
		ExpandTuple(m_parameters, tuple, int_<sizeof...(Args)>());
	}

	// Formatting

	json ToJSON() const override
	{
		json j = FunctionDeclarationBase<R>::ToJSON();
		for (const auto& parameter : m_parameters)
		{
			j["parameters"].push_back(parameter->ToJSON());
		}
		return j;
	}

	// Visitors

	void Accept(Visitor& visitor) override
	{
		if constexpr(std::is_same<R, VoidType>::value)
		{
			visitor.Visit(this);
		}
	}

	void Accept(ConstVisitor& visitor) const override
	{
		if constexpr(std::is_same<R, VoidType>::value)
		{
			visitor.Visit(this);
		}
	}

	void Accept(HierarchicalVisitor& visitor) override
	{
		if constexpr(std::is_same<R, VoidType>::value)
		{
			if (visitor.VisitIn(this))
			{
				for (auto& parameter : m_parameters)
				{
					parameter->Accept(visitor);
				}
			}
			visitor.VisitOut(this);
		}
	}

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		if constexpr(std::is_same<R, VoidType>::value)
		{
			if (visitor.VisitIn(this))
			{
				for (const auto& parameter : m_parameters)
				{
					parameter->Accept(visitor);
				}
			}
			visitor.VisitOut(this);
		}
	}

protected:
	std::vector<VariableDeclaration *> m_parameters;
};

}
