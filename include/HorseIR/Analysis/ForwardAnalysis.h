#pragma once

#include "HorseIR/Analysis/FlowAnalysis.h"

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {

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

		this->m_currentOutSet = this->m_currentInSet;
	}

	void PropagateNext() override
	{
		// Copy the out set to the in set for traversing the next node

		this->m_currentInSet = this->m_currentOutSet;
	}

	void TraverseFunction(const Function *function, const F& initialFlow) override
	{
		// Initialize the input flow set for the function

		this->m_currentInSet = initialFlow;

		// Traverse the parameters and then the body

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

	void TraverseStatements(const std::vector<Statement *>& statements) override
	{
		// Traverse each statement, recording the in/out and propagating sets

		for (const auto statement : statements)                 
		{
			this->m_currentStatement = statement;

			this->SetInSet(statement, this->m_currentInSet);
			statement->Accept(*this);
			this->SetOutSet(statement, this->m_currentOutSet);

			PropagateNext();
		}
		this->m_currentStatement = nullptr;
	}

	void TraverseConditional(const Expression *condition, const BlockStatement *trueBlock, const BlockStatement *elseBlock) override
	{
		// Traverse the true branch and collect sets, store the previous in set
		// for traversing the else branch (if any) after the condition eval

		condition->Accept(*this);
		PropagateNext();

		auto inSet = this->m_currentInSet; // Includes the condition changes

		trueBlock->Accept(*this);
		auto trueOutSet = this->m_currentOutSet;

		if (elseBlock != nullptr)
		{
			// If an else branch is present, analyze, and then merge in the results
			// with the provided merge operation. Reset with in set from condition

			this->m_currentInSet = inSet;

			elseBlock->Accept(*this);
			auto elseOutSet = this->m_currentOutSet;

			this->m_currentOutSet = this->Merge(trueOutSet, elseOutSet);
		}
		else
		{
			// If no else branch is present, merge the in set from the if statement
			// which represents a null else branch (false condition case)

			this->m_currentOutSet = this->Merge(trueOutSet, inSet);
		}
	}

	void Visit(const WhileStatement *whileS) override
	{
		auto [inSet, _] = TraverseLoop(whileS->GetCondition(), whileS->GetBody());
		this->SetInSet(whileS, inSet);
	}

	void Visit(const RepeatStatement *repeatS) override
	{
		auto [inSet, _] = TraverseLoop(repeatS->GetCondition(), repeatS->GetBody());
		this->SetInSet(repeatS, inSet);
	}

	std::tuple<F, F> TraverseLoop(const Expression *condition, const BlockStatement *body) override
	{
		// Save current statement for traversing the condition

		const auto currentStatement = this->m_currentStatement;

		// Create a new loop context, used for storing the continue/break sets

		m_loopContexts.emplace();

		// Store the in set for the loop, not including the condition (we want under all circumstanes)

		auto preSet = this->m_currentInSet;

		this->m_currentStatement = currentStatement;
		condition->Accept(*this);

		// Final sets for the fixed point, used to set the current sets for the statement

		F inSet;
		F outSet;

		do {
			// Store previous out set, used to check for fixed point at the end of the loop

			outSet = this->m_currentOutSet;

			// Reinitialize the break/continue sets for this iteration

			m_loopContexts.top().Initialize();

			// Traverse the body and compute sets for all statements

			PropagateNext();
			body->Accept(*this);

			// Merge all back edge sets (body and continue)

			auto mergedOutSet = this->m_currentOutSet;
			for (const auto& continueSet : m_loopContexts.top().GetContinueSets())
			{
				mergedOutSet = this->Merge(mergedOutSet, continueSet);
			}

			// Create the new in set for the loop, including the previous in set
			// and use this to evaluate the condition and compute the new out set

			this->m_currentInSet = inSet = this->Merge(mergedOutSet, preSet);

			this->m_currentStatement = currentStatement;
			condition->Accept(*this);

		} while (outSet != this->m_currentOutSet);

		// Merge in all break sets to the condition out from the above dowhile loop

		for (const auto& breakSet : m_loopContexts.top().GetBreakSets())
		{
			outSet = this->Merge(outSet, breakSet);
		}

		this->m_currentOutSet = outSet;
		m_loopContexts.pop();

		// Return the new m_currentInSet after the fixed point calculation

		return {inSet, outSet};
	}

	void TraverseBreak(const BreakStatement *breakS) override
	{
		auto context = m_loopContexts.top();
		context.AddBreakSet(this->m_currentInSet);
	}

	void TraverseContinue(const ContinueStatement *continueS) override
	{
		auto context = m_loopContexts.top();
		context.AddContinueSet(this->m_currentInSet);
	}

protected:
	std::stack<LoopContext> m_loopContexts;
};

}
