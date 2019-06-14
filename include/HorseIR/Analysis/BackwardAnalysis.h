#pragma once

#include "HorseIR/Analysis/FlowAnalysis.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {

template<class F>
class BackwardAnalysis : public FlowAnalysis<F>
{
public:
	using SetType = typename FlowAnalysis<F>::SetType;
	using FlowAnalysis<F>::Visit;

	class LoopContext
	{
	public:
		const SetType& GetBreakSet() const { return m_breakSet; }
		void SetBreakSet(const SetType& set) { m_breakSet = set; }

		const SetType& GetContinueSet() const { return m_continueSet; }
		void SetContinueSet(const SetType& set) { m_continueSet = set; }

	private:
		SetType m_breakSet;
		SetType m_continueSet;
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

	void TraverseStatements(const std::vector<Statement *>& statements) override
	{
		// Traverse each statement, recording the in/out and propagating sets

		for (auto it = statements.rbegin(); it != statements.rend(); ++it)
		{
			auto statement = *it;

			this->SetOutSet(statement, this->m_currentOutSet);
			statement->Accept(*this);
			this->SetInSet(statement, this->m_currentInSet);

			PropagateNext();
		}
	}

	void TraverseConditional(const Expression *condition, const BlockStatement *trueBlock, const BlockStatement *elseBlock) override
	{
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

			this->m_currentInSet = Merge(trueInSet, elseInSet);
		}
		else
		{
			// If no else branch is present, merge the in set from the if statement
			// which represents a null else branch (false condition case)

			this->m_currentInSet = Merge(trueInSet, outSet);
		}

		PropagateNext();
		condition->Accept(*this);
	}

	//TODO: break and continue/loop contexts needs updating for backwards (split)

	std::tuple<SetType, SetType> TraverseLoop(const Expression *condition, const BlockStatement *body) override
	{
		// Create a new loop context, used for storing the continue/break sets

		m_loopContexts.emplace();

		// Save out set for the loop breaks, as well as the skip edge coming from the condition

		auto outSet = this->m_currentOutSet;
		m_loopContexts.top().SetBreakSet(outSet);

		// Evaluate the condition after break setup (breaks do not evaluate the condition)

		condition->Accept(*this);
		PropagateNext();

		// Save the current in set (loop 0 iterations)
		
		SetType inSet = this->m_currentInSet;

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

			this->m_currentInSet = Merge(outSet, this->m_currentInSet);

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

	// Each analysis must provide its own merge operation

	virtual SetType Merge(const SetType& s1, const SetType& s2) const = 0;

protected:
	std::stack<LoopContext> m_loopContexts;
};

}
