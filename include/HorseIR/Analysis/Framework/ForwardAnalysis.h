#pragma once

#include "HorseIR/Analysis/Framework/FlowAnalysis.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {
namespace Analysis {

template<class F>
class ForwardAnalysis : public FlowAnalysis<F>
{
public:
	using FlowAnalysis<F>::FlowAnalysis;
	using FlowAnalysis<F>::Visit;

	class LoopContext
	{
	public:
		const std::vector<F>& GetBreakSets() const { return m_breakSets; }
		void AddBreakSet(const F& set)
		{
			m_breakSets.push_back(set);
		}

		const std::vector<F>& GetContinueSets() const { return m_continueSets; }
		void AddContinueSet(const F& set)
		{
			m_continueSets.push_back(set);
		}

		void Initialize()
		{
			m_breakSets.clear();
			m_continueSets.clear();
		}

	private:
		std::vector<F> m_breakSets;
		std::vector<F> m_continueSets;
	};

	void Visit(const Node *node) override
	{
		// Default action, propagate the set forward with no changes
	}

	void TraverseFunction(const Function *function, const F& initialFlow) override
	{
		// Initialize the input flow set for the function

		this->m_currentSet = initialFlow;

		// Traverse the parameters and then the body

		for (const auto& parameter : function->GetParameters())
		{
			parameter->Accept(*this);
		}
		for (const auto& returnType : function->GetReturnTypes())
		{
			returnType->Accept(*this);
		}
		TraverseStatements(function->GetStatements());
	}

	void TraverseStatements(const std::vector<const Statement *>& statements) override
	{
		// Traverse each statement, recording the in/out and propagating sets

		for (const auto statement : statements)                 
		{
			this->m_currentStatement = statement;

			if (this->CollectInSets())
			{
				this->SetInSet(statement, this->m_currentSet);
			}
			statement->Accept(*this);
			if (this->CollectOutSets())
			{
				this->SetOutSet(statement, this->m_currentSet);
			}
		}
		this->m_currentStatement = nullptr;

		this->m_endSet = this->m_currentSet;
	}

	void TraverseConditional(const Expression *condition, const BlockStatement *trueBlock, const BlockStatement *elseBlock) override
	{
		// Traverse the true branch and collect sets, store the previous in set
		// for traversing the else branch (if any) after the condition eval

		condition->Accept(*this);

		auto inSet = this->m_currentSet; // Includes the condition changes

		trueBlock->Accept(*this);
		auto trueOutSet = this->m_currentSet;

		if (elseBlock != nullptr)
		{
			// If an else branch is present, analyze, and then merge in the results
			// with the provided merge operation. Reset with in set from condition

			this->m_currentSet = inSet;

			elseBlock->Accept(*this);
			auto elseOutSet = this->m_currentSet;

			this->m_currentSet = this->Merge(trueOutSet, elseOutSet);
		}
		else
		{
			// If no else branch is present, merge the in set from the if statement
			// which represents a null else branch (false condition case)

			this->m_currentSet = this->Merge(trueOutSet, inSet);
		}
	}

	void Visit(const WhileStatement *whileS) override
	{
		auto [inSet, _] = TraverseLoop(whileS->GetCondition(), whileS->GetBody());

		if (this->CollectInSets())
		{
			this->SetInSet(whileS, inSet);
		}
	}

	void Visit(const RepeatStatement *repeatS) override
	{
		auto [inSet, _] = TraverseLoop(repeatS->GetCondition(), repeatS->GetBody());

		if (this->CollectInSets())
		{
			this->SetInSet(repeatS, inSet);
		}
	}

	std::tuple<F, F> TraverseLoop(const Expression *condition, const BlockStatement *body) override
	{
		// Save current statement for traversing the condition

		const auto currentStatement = this->m_currentStatement;

		// Create a new loop context, used for storing the continue/break sets

		m_loopContexts.emplace();

		// Store the in set for the loop, not including the condition (we want under all circumstanes)

		auto preSet = this->m_currentSet;

		this->m_currentStatement = currentStatement;
		condition->Accept(*this);

		// Final sets for the fixed point, used to set the current sets for the statement

		F inSet;
		F outSet;

		do {
			// Store previous out set, used to check for fixed point at the end of the loop

			outSet = this->m_currentSet;

			// Reinitialize the break/continue sets for this iteration

			m_loopContexts.top().Initialize();

			// Traverse the body and compute sets for all statements

			body->Accept(*this);

			// Merge all back edge sets (body and continue)

			auto mergedOutSet = this->m_currentSet;
			for (const auto& continueSet : m_loopContexts.top().GetContinueSets())
			{
				mergedOutSet = this->Merge(mergedOutSet, continueSet);
			}

			// Create the new in set for the loop, including the previous in set
			// and use this to evaluate the condition and compute the new out set

			this->m_currentSet = inSet = this->Merge(mergedOutSet, preSet);

			this->m_currentStatement = currentStatement;
			condition->Accept(*this);

		} while (outSet != this->m_currentSet);

		// Merge in all break sets to the condition out from the above dowhile loop

		for (const auto& breakSet : m_loopContexts.top().GetBreakSets())
		{
			outSet = this->Merge(outSet, breakSet);
		}

		this->m_currentSet = outSet;
		m_loopContexts.pop();

		// Return the new m_currentSet after the fixed point calculation

		return {inSet, outSet};
	}

	void TraverseBreak(const BreakStatement *breakS) override
	{
		auto context = m_loopContexts.top();
		context.AddBreakSet(this->m_currentSet);
	}

	void TraverseContinue(const ContinueStatement *continueS) override
	{
		auto context = m_loopContexts.top();
		context.AddContinueSet(this->m_currentSet);
	}

	const F& GetEndSet() const { return m_endSet; }

protected:
	std::stack<LoopContext> m_loopContexts;

	F m_endSet;
};

}
}
