#pragma once

#include <string>
#include <sstream>
#include <vector>

#include "HorseIR/Analysis/Framework/StatementAnalysis.h"
#include "HorseIR/Analysis/Framework/StatementAnalysisPrinter.h"
#include "HorseIR/Traversal/ConstVisitor.h"
#include "HorseIR/Tree/Tree.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

#include "Libraries/robin_hood.h"

namespace HorseIR {
namespace Analysis {

template<class F>
class FlowAnalysis : public ConstVisitor, public StatementAnalysis
{
public:
	using ConstVisitor::Visit;

	FlowAnalysis(const std::string& name, const std::string& shortName, const Program *program) : m_name(name), m_shortName(shortName), m_program(program) {}

	void Analyze(const Function *function)
	{
		m_analysisTime = Utils::Chrono::Start(m_name + " '" + function->GetName() + "'");

		auto timeInit_start = Utils::Chrono::Start(m_name + " initial flow");
		auto initialFlow = InitialFlow();
		Utils::Chrono::End(timeInit_start);

		Analyze(function, initialFlow);
	}

	void Analyze(const Function *function, const F& initialFlow)
	{
		// Clear sets and traverse the function, timing results

		auto& functionName = function->GetName();
		if (m_analysisTime == nullptr)
		{
			m_analysisTime = Utils::Chrono::Start(m_name + " '" + functionName + "'");
		}

		m_functionTime = Utils::Chrono::Start(m_name + " '" + functionName + "' body");

		this->m_currentSet.clear();

		TraverseFunction(function, initialFlow);

		// Print results if needed

		if (Utils::Options::IsFrontend_PrintAnalysis(m_shortName, functionName))
		{
			PrintResults(function);
		}

		Utils::Chrono::End(m_functionTime);
		Utils::Chrono::End(m_analysisTime);
	}

	void PrintResults(const Function *function)
	{
		Utils::Logger::LogInfo(m_name + " '" + function->GetName() + "'");

		auto string = StatementAnalysisPrinter::PrettyString(*this, function);
		Utils::Logger::LogInfo(string, 0, true, Utils::Logger::NoPrefix);
	}

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
		for (const auto& operand : returnS->GetOperands())
		{
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
			argument->Accept(*this);
		}
	}

	void Visit(const CastExpression *cast) override
	{
		cast->GetExpression()->Accept(*this);
		cast->GetCastType()->Accept(*this);
	}

	virtual void TraverseFunction(const Function *function, const F& initialFlow) = 0;
	virtual void TraverseStatements(const std::vector<const Statement *>& statements) = 0;

	virtual void TraverseConditional(const Expression *condition, const BlockStatement *trueBlock, const BlockStatement *falseBlock) = 0;
	virtual std::tuple<F, F> TraverseLoop(const Expression *condition, const BlockStatement *body) = 0;

	virtual void TraverseBreak(const BreakStatement *breakS) = 0;
	virtual void TraverseContinue(const ContinueStatement *continueS) = 0;

	// Accessors

	const F& GetInSet(const Statement *statement) const { return m_inSets.at(statement); }
	const F& GetOutSet(const Statement *statement) const { return m_outSets.at(statement); }

	// Options

	bool CollectInSets() const { return m_collectInSets; }
	bool CollectOutSets() const { return m_collectOutSets; }

	void SetCollectInSets(bool collectInSets) { m_collectInSets = collectInSets; }
	void SetCollectOutSets(bool collectOutSets) { m_collectOutSets = collectOutSets; }

	const std::string& GetName() const { return m_name; }
	const std::string& GetShortName() const { return m_shortName; }

	// Formatting

	std::string DebugString(const Statement *statement, unsigned indent = 0) const override
	{
		std::string indentString(indent * Utils::Logger::IndentSize, ' ');

		std::stringstream string;
		if (m_collectInSets)
		{
			string << indentString << "In: {" << std::endl;
			GetInSet(statement).Print(string, indent + 1);
			string << std::endl;
			string << indentString << "}";

			if (m_collectOutSets)
			{
				string << std::endl;
			}
		}

		if (m_collectOutSets)
		{
			string << indentString << "Out: {" << std::endl;
			GetOutSet(statement).Print(string, indent + 1);
			string << std::endl;
			string << indentString << "}";
		}

		return string.str();
	}

	// Each analysis must provide its initial flows and merge operation

	virtual F InitialFlow() const = 0;
	virtual F Merge(const F& s1, const F& s2) const = 0;

protected:
	void SetInSet(const Statement *statement, const F& set) { m_inSets.insert_or_assign(statement, set); }
	void SetOutSet(const Statement *statement, const F& set) { m_outSets.insert_or_assign(statement, set); }

	const Program *m_program = nullptr;
	const Statement *m_currentStatement = nullptr;

	robin_hood::unordered_map<const Statement *, F> m_inSets;
	robin_hood::unordered_map<const Statement *, F> m_outSets;

	bool m_collectInSets = true;
	bool m_collectOutSets = true;

	F m_currentSet;

	const Utils::Chrono::SpanTiming *m_analysisTime = nullptr;
	const Utils::Chrono::SpanTiming *m_functionTime = nullptr;

	std::string m_name;
	std::string m_shortName;
};

}
}
