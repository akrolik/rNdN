#pragma once

#include "HorseIR/Analysis/FlowAnalysis.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {

template<class F>
class BackwardAnalysis : public FlowAnalysis<F>
{
public:
	using FlowAnalysis<F>::FlowAnalysis;
	using FlowAnalysis<F>::Visit;

	class LoopContext
	{
	public:
		const F& GetBreakSet() const { return m_breakSet; }
		void SetBreakSet(const F& set) { m_breakSet = set; }

		const F& GetContinueSet() const { return m_continueSet; }
		void SetContinueSet(const F& set) { m_continueSet = set; }

	private:
		F m_breakSet;
		F m_continueSet;
	};

	void Visit(const Node *node) override
	{
		// Default action, propagate the set forward with no changes

		this->m_currentInSet = this->m_currentOutSet;
	}

	void PropagateNext() override
	{
		// Copy the out set to the in set for traversing the next node

		this->m_currentOutSet = this->m_currentInSet;
	}

	void TraverseFunction(const Function *function, const F& initialFlow) override
	{
		// Initialize the input flow set for the function

		this->m_currentOutSet = initialFlow;

		// Traverse the body and then the parameters

		TraverseStatements(function->GetStatements());
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
	}

	void TraverseStatements(const std::vector<Statement *>& statements) override
	{
		// Traverse each statement, recording the in/out and propagating sets

		for (auto it = statements.rbegin(); it != statements.rend(); ++it)
		{
			auto statement = *it;

			this->m_currentStatement = statement;

			this->SetOutSet(statement, this->m_currentOutSet);
			statement->Accept(*this);
			this->SetInSet(statement, this->m_currentInSet);

			PropagateNext();
		}
		this->m_currentStatement = nullptr;
	}

	void TraverseConditional(const Expression *condition, const BlockStatement *trueBlock, const BlockStatement *elseBlock) override
	{
		// Save current statement for traversing the condition

		const auto currentStatement = this->m_currentStatement;

		// Traverse the true branch and collect sets, store the previous out set
		// for traversing the else branch (if any)

		auto outSet = this->m_currentOutSet;

		trueBlock->Accept(*this);
		auto trueInSet = this->m_currentInSet;

		if (elseBlock != nullptr)
		{
			// If an else branch is present, analyze, and then merge in the results
			// with the provided merge operation. Reset with in set from condition

			this->m_currentOutSet = outSet;

			elseBlock->Accept(*this);
			auto elseInSet = this->m_currentInSet;

			this->m_currentInSet = this->Merge(trueInSet, elseInSet);
		}
		else
		{
			// If no else branch is present, merge the in set from the if statement
			// which represents a null else branch (false condition case)

			this->m_currentInSet = this->Merge(trueInSet, outSet);
		}

		PropagateNext();

		this->m_currentStatement = currentStatement;
		condition->Accept(*this);
	}

	std::tuple<F, F> TraverseLoop(const Expression *condition, const BlockStatement *body) override
	{
		// Save current statement for traversing the condition

		const auto currentStatement = this->m_currentStatement;

		// Create a new loop context, used for storing the continue/break sets

		m_loopContexts.emplace();

		// Save out set for the loop breaks, as well as the skip edge coming from the condition

		auto outSet = this->m_currentOutSet;
		m_loopContexts.top().SetBreakSet(outSet);

		// Evaluate the condition after break setup (breaks do not evaluate the condition)

		this->m_currentStatement = currentStatement;
		condition->Accept(*this);

		PropagateNext();

		// Information for computing the fixed point
		
		F inSet;

		do
		{
			// Store previous in set for the fixed point computation

			inSet = this->m_currentInSet;

			// Setup next iteration, propagating from the in of the previous statement (either body or exit)

			PropagateNext();

			// Save the set from the latter iteration for continues

			m_loopContexts.top().SetContinueSet(this->m_currentOutSet);

			body->Accept(*this);
			PropagateNext();

			this->m_currentInSet = this->Merge(outSet, this->m_currentInSet);

			this->m_currentStatement = currentStatement;
			condition->Accept(*this);

		} while (inSet != this->m_currentInSet);

		m_loopContexts.pop();

		// Return the new m_currentInSet after the fixed point calculation

		return {inSet, outSet};
	}

	void TraverseBreak(const BreakStatement *breakS) override
	{
		auto outSet = m_loopContexts.top().GetBreakSet();
		this->SetOutSet(breakS, outSet);
	}

	void TraverseContinue(const ContinueStatement *continueS) override
	{
		auto outSet = m_loopContexts.top().GetContinueSet();
		this->SetOutSet(continueS, outSet);
	}

protected:
	std::stack<LoopContext> m_loopContexts;
};

}
