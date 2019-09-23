#pragma once

#include <sstream>
#include <stack>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "HorseIR/Traversal/ConstVisitor.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {

template<typename T>
struct FlowAnalysisPointerValue
{
	struct Equals
	{
		 bool operator()(const T *val1, const T *val2) const
		 {
			 return val1 == val2;
		 }
	};
};

template<typename T>
struct FlowAnalysisValue
{
	struct Equals
	{
		 bool operator()(const T *val1, const T *val2) const
		 {
			 return *val1 == *val2;
		 }
	};
};

template<typename T>
struct FlowAnalysisSet : public std::unordered_set<const typename T::Type *, typename T::Hash, typename T::Equals>
{
	void Print(std::ostream& os, unsigned int level = 0) const
	{
		for (unsigned int i = 0; i < level; ++i)
		{
			os << '\t';
		}
		bool first = true;
		for (const auto& val : *this)
		{
			if (!first)
			{
				os << ", ";
			}
			first = false;
			T::Print(os, val);
		}
	}

	bool operator==(const FlowAnalysisSet<T>& other) const
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

	bool operator!=(const FlowAnalysisSet<T>& other) const
	{
		return !(*this == other);
	}
};

template<typename K, typename V>
struct FlowAnalysisMap : public std::unordered_map<const typename K::Type *, const typename V::Type *, typename K::Hash, typename K::Equals>
{
	void Print(std::ostream& os, unsigned int level = 0) const
	{
		bool first = true;
		for (const auto& pair : *this)
		{
			if (!first)
			{
				os << std::endl;
			}
			first = false;
			for (unsigned int i = 0; i < level; ++i)
			{
				os << '\t';
			}
			K::Print(os, pair.first);
			os << "->";
			V::Print(os, pair.second);
		}
	}

	bool operator==(const FlowAnalysisMap<K, V>& other) const
	{
		if (this->size() != other.size())
		{
			return false;
		}

		for (auto it = this->begin(); it != this->end(); ++it)
		{
			auto y = other.find(it->first);
			if (y == other.end() || !typename V::Equals()(y->second, it->second))
			{
				return false;
			}

		}
		return true;
	}

	bool operator!=(const FlowAnalysisMap<K, V>& other)const
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
	using ConstVisitor::Visit;

	FlowAnalysis(const Program *program) : m_program(program) {}

	void Analyze(const Function *function)
	{
		// Clear sets and traverse the function

		this->m_currentInSet.clear();
		this->m_currentOutSet.clear();

		TraverseFunction(function);
	}

	virtual void PropagateNext() = 0;

	virtual void Visit(const Node *node) = 0;

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
		TraverseLoop(whileS->GetCondition(), whileS->GetBody());
	}

	void Visit(const RepeatStatement *repeatS) override
	{
		TraverseLoop(repeatS->GetCondition(), repeatS->GetBody());
	}

	void Visit(const BlockStatement *blockS) override
	{
		TraverseStatements(blockS->GetStatements());
	}

	void Visit(const ReturnStatement *returnS) override
	{
		bool first = true;
		for (const auto& operand : returnS->GetOperands())
		{
			if (!first)
			{
				PropagateNext();
			}
			first = false;
			operand->Accept(*this);
		}
	}

	void Visit(const BreakStatement *breakS) override
	{
		TraverseBreak(breakS);
	}

	void Visit(const ContinueStatement *continueS) override
	{
		TraverseContinue(continueS);
	}

	void Visit(const CallExpression *call) override
	{
		call->GetFunctionLiteral()->Accept(*this);
		for (const auto& argument : call->GetArguments())
		{
			PropagateNext();
			argument->Accept(*this);
		}
	}

	void Visit(const CastExpression *cast) override
	{
		cast->GetExpression()->Accept(*this);
		PropagateNext();
		cast->GetCastType()->Accept(*this);
	}

	virtual void TraverseFunction(const Function *function) = 0;
	virtual void TraverseStatements(const std::vector<Statement *>& statements) = 0;

	virtual void TraverseConditional(const Expression *condition, const BlockStatement *trueBlock, const BlockStatement *falseBlock) = 0;
	virtual std::tuple<F, F> TraverseLoop(const Expression *condition, const BlockStatement *body) = 0;

	virtual void TraverseBreak(const BreakStatement *breakS) = 0;
	virtual void TraverseContinue(const ContinueStatement *continueS) = 0;

	const F& GetInSet(const Statement *statement) const { return m_inSets.at(statement); }
	const F& GetOutSet(const Statement *statement) const { return m_outSets.at(statement); }

protected:
	void SetInSet(const Statement *statement, const F& set) { m_inSets.insert_or_assign(statement, set); }
	void SetOutSet(const Statement *statement, const F& set) { m_outSets.insert_or_assign(statement, set); }

	const Program *m_program = nullptr;
	const Statement *m_currentStatement = nullptr;

	std::unordered_map<const Statement *, F> m_inSets;
	std::unordered_map<const Statement *, F> m_outSets;

	F m_currentInSet;
	F m_currentOutSet;
};

}
