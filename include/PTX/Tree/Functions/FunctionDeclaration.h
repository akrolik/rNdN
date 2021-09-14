#pragma once

#include "PTX/Tree/Functions/FunctionDeclarationBase.h"

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Tuple.h"
#include "PTX/Tree/Type.h"
#include "PTX/Tree/Declarations/Declaration.h"
#include "PTX/Tree/Declarations/VariableDeclaration.h"

namespace PTX {

DispatchInterface_Space(FunctionDeclaration)

template<class RT, class RS = ParameterSpace, bool Assert = true>
class FunctionDeclaration : DispatchInherit(FunctionDeclaration), public FunctionDeclarationBase<RT, RS>
{
public:
	REQUIRE_TYPE_PARAM(FunctionDeclaration,
		REQUIRE_BASE(RT, ScalarType) || REQUIRE_EXACT(RT, VoidType)
	);
	REQUIRE_SPACE_PARAM(FunctionDeclaration,
		REQUIRE_EXACT(RS, RegisterSpace, ParameterSpace)
	);

	using FunctionDeclarationBase<RT, RS>::FunctionDeclarationBase;

	// Properties

	std::vector<const VariableDeclaration *> GetParameters() const
	{
		return { std::begin(m_parameters), std::end(m_parameters) };
	}
	std::vector<VariableDeclaration *>& GetParameters() { return m_parameters; }

	template<class T, class S>
	std::enable_if_t<REQUIRE_EXACT(S, RegisterSpace) || REQUIRE_BASE(S, ParameterSpace), void>
	AddParameter(TypedVariableDeclaration<T, S> *parameter)
	{
		m_parameters.push_back(parameter);
	}

	// Formatting

	json ToJSON() const override
	{
		json j = FunctionDeclarationBase<RT, RS>::ToJSON();
		for (const auto& parameter : m_parameters)
		{
			j["parameters"].push_back(parameter->ToJSON());
		}
		return j;
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

	void Accept(HierarchicalVisitor& visitor) override
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

	void Accept(ConstHierarchicalVisitor& visitor) const override
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

	void Accept(FunctionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstFunctionVisitor& visitor) const { visitor.Visit(this); }

protected:
	DispatchMember_Type(RT);
	DispatchMember_Space(RS);

	std::vector<VariableDeclaration *> m_parameters;
};

template<class R, typename... Args>
class FunctionDeclaration<R(Args...)> : public FunctionDeclaration<typename R::VariableType, typename R::VariableSpace>
{
public:
	REQUIRE_SPACE_PARAM(FunctionDeclaration,
		is_all<REQUIRE_EXACT(typename Args::VariableSpace, RegisterSpace) || REQUIRE_BASE(typename Args::VariableSpace, ParameterSpace)...>::value
	);

	using ReturnDeclarationType = TypedVariableDeclaration<typename R::VariableType, typename R::VariableSpace>;
	using Signature = R(Args...);

	FunctionDeclaration() {}
	FunctionDeclaration(const std::string& name, ReturnDeclarationType *ret,
			TypedVariableDeclaration<typename Args::VariableType, typename Args::VariableSpace>* ...parameters,
			Declaration::LinkDirective linkDirective = Declaration::LinkDirective::None
	) : FunctionDeclaration<typename R::VariableType, typename R::VariableSpace>(name, ret, linkDirective)
	{
		auto tuple = std::make_tuple(parameters...);
		ExpandTuple(this->m_parameters, tuple, int_<sizeof...(Args)>());
	}

	// Properties

	void SetParameters(TypedVariableDeclaration<typename Args::VariableType, typename Args::VariableSpace>* ...parameters)
	{
		auto tuple = std::make_tuple(parameters...);
		ExpandTuple(this->m_parameters, tuple, int_<sizeof...(Args)>());
	}
};

}
