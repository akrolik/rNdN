#pragma once

#include <stack>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "HorseIR/Traversal/ConstVisitor.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {

template<class F>
struct FlowAnalysisSet : public std::unordered_set<const F *, typename F::Hash, typename F::Equals>
{
	bool operator==(const FlowAnalysisSet<F>& other) const
	{
		if (this->size() != other.size())
		{
			return false;
		}

		for (auto it = this->begin(); it != this->end(); ++it)
		{
			if (other.find(*it) == other.end())
			{
				return false;
			}
		}
		return true;
	}

	bool operator!=(const FlowAnalysisSet<F>& other) const
	{
		return !(*this == other);
	}
};

template<class F>
class FlowAnalysisPrinter;

template<class F>
class FlowAnalysis : public ConstVisitor
{
public:
	friend class FlowAnalysisPrinter<F>;

	using SetType = FlowAnalysisSet<F>;
	using ConstVisitor::Visit;

	void Analyze(const Function *function)
	{
		m_currentInSet.clear();
		m_currentOutSet.clear();

		for (const auto& parameter : function->GetParameters())
		{
			parameter->Accept(*this);
			PropagateNext();
		}
		for (const auto& returnType : function->GetReturnTypes())
		{
			returnType->Accept(*this);
			PropagateNext();
		}
		TraverseStatements(function->GetStatements());
	}

	void Visit(const VariableDeclaration *declaration) override
	{
		declaration->GetType()->Accept(*this);
	}

	void Visit(const DeclarationStatement *declarationS) override
	{
		declarationS->GetDeclaration()->Accept(*this);
	}

	void Visit(const AssignStatement *assignS) override
	{
		for (const auto& target : assignS->GetTargets())
		{
			target->Accept(*this);
			PropagateNext();
		}
		assignS->GetExpression()->Accept(*this);
	}

	void Visit(const ExpressionStatement *expressionS) override
	{
		expressionS->GetExpression()->Accept(*this);
	}

	void Visit(const IfStatement *ifS) override
	{
		TraverseConditional(ifS->GetCondition(), ifS->GetTrueBlock(), ifS->GetElseBlock());
	}

	void Visit(const WhileStatement *whileS) override
	{
		auto [inSet, outSet] = TraverseLoop(whileS->GetCondition(), whileS->GetBody());
		//TODO: this is only needed for forward analysis
		SetInSet(whileS, inSet);
	}

	void Visit(const RepeatStatement *repeatS) override
	{
		auto [inSet, outSet] = TraverseLoop(repeatS->GetCondition(), repeatS->GetBody());
		//TODO: this is only needed for forward analysis
		SetInSet(repeatS, inSet);
	}

	void Visit(const BlockStatement *blockS) override
	{
		TraverseStatements(blockS->GetStatements());
	}

	void Visit(const ReturnStatement *returnS) override
	{
		for (const auto& operand : returnS->GetOperands())
		{
			operand->Accept(*this);
			PropagateNext();
		}
	}

	void Visit(const BreakStatement *breakS) override
	{
		TraverseBreak(breakS);
		Visit(static_cast<const Statement *>(breakS));
	}

	void Visit(const ContinueStatement *continueS) override
	{
		TraverseContinue(continueS);
		Visit(static_cast<const Statement *>(continueS));
	}

	void Visit(const CallExpression *call) override
	{
		call->GetFunctionLiteral()->Accept(*this);
		PropagateNext();
		for (const auto& argument : call->GetArguments())
		{
			argument->Accept(*this);
			PropagateNext();
		}
	}

	void Visit(const CastExpression *cast) override
	{
		cast->GetExpression()->Accept(*this);
		PropagateNext();
		cast->GetCastType()->Accept(*this);
	}

	virtual void PropagateNext() = 0;

	virtual void TraverseStatements(const std::vector<Statement *>& statements) = 0;

	virtual void TraverseConditional(const Expression *condition, const BlockStatement *trueBlock, const BlockStatement *falseBlock) = 0;
	virtual std::tuple<SetType, SetType> TraverseLoop(const Expression *condition, const BlockStatement *body) = 0;

	virtual void TraverseBreak(const BreakStatement *breakS) = 0;
	virtual void TraverseContinue(const ContinueStatement *continueS) = 0;

protected:
	void SetInSet(const Statement *statement, const SetType& set)
	{
		m_inSets.insert_or_assign(statement, set);
	}

	const SetType& GetInSet(const Statement *statement) const
	{
		return m_inSets.at(statement);
	}

	void SetOutSet(const Statement *statement, const SetType& set)
	{
		m_outSets.insert_or_assign(statement, set);
	}

	const SetType& GetOutSet(const Statement *statement) const
	{
		return m_outSets.at(statement);
	}

	std::unordered_map<const Statement *, SetType> m_inSets;
	std::unordered_map<const Statement *, SetType> m_outSets;

	SetType m_currentInSet;
	SetType m_currentOutSet;
};

}
