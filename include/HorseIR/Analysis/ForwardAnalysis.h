#pragma once

#include <stack>
#include <unordered_set>
#include <unordered_map>
#include <vector>

#include "HorseIR/Traversal/ConstVisitor.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {

template<class F>
struct AnalysisSet : public std::unordered_set<const F *, typename F::Hash, typename F::Equals>
{
	bool operator==(const AnalysisSet<F>& other) const
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

	bool operator!=(const AnalysisSet<F>& other) const
	{
		return !(*this == other);
	}
};

template<class F>
class AnalysisPrinter;

template<class F>
class ForwardAnalysis : public ConstVisitor
{
public:
	friend class AnalysisPrinter<F>;

	using SetType = AnalysisSet<F>;

	class LoopContext
	{
	public:
		const std::vector<SetType>& GetBreakSets() const { return m_breakSets; }
		void AddBreakSet(const SetType& set)
		{
			m_breakSets.push_back(set);
		}

		const std::vector<SetType>& GetContinueSets() const { return m_continueSets; }
		void AddContinueSet(const SetType& set)
		{
			m_continueSets.push_back(set);
		}

		void Initialize()
		{
			m_breakSets.clear();
			m_continueSets.clear();
		}

	private:
		std::vector<SetType> m_breakSets;
		std::vector<SetType> m_continueSets;
	};

	void Analyze(const Function *function)
	{
		m_currentInSet.clear();
		m_currentOutSet.clear();

		for (const auto& parameter : function->GetParameters())
		{
			parameter->Accept(*this);
		}
		for (const auto& returnType : function->GetReturnTypes())
		{
			returnType->Accept(*this);
		}
		for (const auto statement : function->GetStatements())
		{
			SetInSet(statement, m_currentInSet);
			statement->Accept(*this);
			SetOutSet(statement, m_currentOutSet);

			m_currentInSet = m_currentOutSet;
		}
	}

	void Visit(const Node *node) override
	{
		m_currentOutSet = m_currentInSet;
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
			m_currentInSet = m_currentOutSet;
		}
		assignS->GetExpression()->Accept(*this);
	}

	void Visit(const ExpressionStatement *expressionS) override
	{
		expressionS->GetExpression()->Accept(*this);
	}

	void Visit(const IfStatement *ifS) override
	{
		// Traverse the true branch and collect sets, store the previous in set
		// for traversing the else branch (if any) after the condition eval

		ifS->GetCondition()->Accept(*this);
		auto inSet = m_currentOutSet; // Includes the condition changes

		auto trueBlock = ifS->GetTrueBlock();
		trueBlock->Accept(*this);
		auto trueOutSet = m_currentOutSet;

		if (ifS->HasElseBranch())
		{
			// If an else branch is present, analyze, and then merge in the results
			// with the provided merge operation. Reset with in set from condition

			m_currentInSet = inSet;

			auto elseBlock = ifS->GetElseBlock();
			elseBlock->Accept(*this);
			auto elseOutSet = m_currentOutSet;

			m_currentOutSet = Merge(trueOutSet, elseOutSet);
		}
		else
		{
			// If no else branch is present, merge the in set from the if statement
			// which represents a null else branch (false condition case)

			m_currentOutSet = Merge(trueOutSet, inSet);
		}
	}

	void Visit(const WhileStatement *whileS) override
	{
		auto inSet = VisitLoop(whileS->GetCondition(), whileS->GetBody());
		SetInSet(whileS, inSet);
	}

	void Visit(const RepeatStatement *repeatS) override
	{
		auto inSet = VisitLoop(repeatS->GetCondition(), repeatS->GetBody());
		SetInSet(repeatS, inSet);
	}

	SetType VisitLoop(const Expression *condition, const BlockStatement *body)
	{
		// Create a new loop context, used for storing the continue/break sets

		m_loopContexts.emplace();

		// Store the in set for the loop, not including the condition (we want under all circumstanes)

		auto preSet = m_currentInSet;
		condition->Accept(*this);

		// Final sets for the fixed point, used to set the current sets for the statement

		SetType inSet;
		SetType outSet;

		do {
			// Store previous out set, used to check for fixed point at the end of the loop

			outSet = m_currentOutSet;

			// Reinitialize the break/continue sets for this iteration

			m_loopContexts.top().Initialize();

			// Traverse the body and compute sets for all statements

			m_currentInSet = m_currentOutSet;
			body->Accept(*this);

			// Merge all back edge sets (body and continue)

			auto mergedOutSet = m_currentOutSet;
			for (const auto& continueSet : m_loopContexts.top().GetContinueSets())
			{
				mergedOutSet = Merge(mergedOutSet, continueSet);
			}

			// Create the new in set for the loop, including the previous in set
			// and use this to evaluate the condition and compute the new out set

			m_currentInSet = inSet = Merge(mergedOutSet, preSet);
			condition->Accept(*this);

		} while (outSet != m_currentOutSet);

		// Merge in all break sets to the condition out from the above dowhile loop

		for (const auto& breakSet : m_loopContexts.top().GetBreakSets())
		{
			outSet = Merge(outSet, breakSet);
		}

		m_currentOutSet = outSet;
		m_loopContexts.pop();

		// Return the new m_currentInSet after the fixed point calculation

		return inSet;
	}

	void Visit(const BlockStatement *blockS) override
	{
		// Traverse each statement, recording the in/out and propagating sets

		for (const auto statement : blockS->GetStatements())                 
		{
			SetInSet(statement, m_currentInSet);
			statement->Accept(*this);
			SetOutSet(statement, m_currentOutSet);

			m_currentInSet = m_currentOutSet;
		}
	}

	void Visit(const ReturnStatement *returnS) override
	{
		for (const auto& operand : returnS->GetOperands())
		{
			operand->Accept(*this);

			m_currentInSet = m_currentOutSet;
		}
	}

	void Visit(const BreakStatement *breakS) override
	{
		auto context = m_loopContexts.top();
		context.AddBreakSet(m_currentInSet);

		m_currentOutSet = m_currentInSet;
	}

	void Visit(const ContinueStatement *continueS) override
	{
		auto context = m_loopContexts.top();
		context.AddContinueSet(m_currentInSet);
		
		m_currentOutSet = m_currentInSet;
	}

	void Visit(const CallExpression *call) override
	{
		call->GetFunctionLiteral()->Accept(*this);
		for (const auto& argument : call->GetArguments())
		{
			m_currentInSet = m_currentOutSet;
			argument->Accept(*this);
		}
	}

	void Visit(const CastExpression *cast) override
	{
		cast->GetExpression()->Accept(*this);
		m_currentInSet = m_currentOutSet;
		cast->GetCastType()->Accept(*this);
	}

	// Each analysis must provide its own merge operation

	virtual SetType Merge(const SetType& s1, const SetType& s2) const = 0;

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

	std::stack<LoopContext> m_loopContexts;
	SetType m_currentInSet;
	SetType m_currentOutSet;
};

}
