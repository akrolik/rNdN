#pragma once

#include <sstream>
#include <string>

#include "PTX/Traversal/ConstVisitor.h"
#include "PTX/Tree/Tree.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace PTX {
namespace Analysis {

template<class F>
class FlowInsensitiveAnalysis : public ConstVisitor
{
public:
	FlowInsensitiveAnalysis(const std::string& name, const std::string& shortName) : m_name(name), m_shortName(shortName) {}

	void Analyze(const FunctionDefinition<VoidType> *function)
	{
		auto& functionName = function->GetName();
		auto analysisTime = Utils::Chrono::Start(m_name + " '" + functionName + "'");

		// Clear sets and traverse the function, timing results

		this->m_analysisSet.clear();

		const auto cfg = function->GetControlFlowGraph();
		for (const auto block : cfg->GetNodes())
		{
			block->Accept(*this);
		}

		// Print results if needed

		if (Utils::Options::IsBackend_PrintAnalysis(m_shortName, functionName))
		{
			PrintResults(function);
		}

		Utils::Chrono::End(analysisTime);
	}

	// Analysis framework

	void Visit(const BasicBlock *block) override
	{
		TraverseStatements(block->GetStatements());
	}

	void Visit(const Node *node) override
	{
		// Do nothing
	}

	void TraverseStatements(const std::vector<const Statement *>& statements)
	{
		for (const auto& statement : statements)
		{
			statement->Accept(*this);
		}
	}

	void Visit(const BlockStatement *statement) override
	{
		TraverseStatements(statement->GetStatements());
	}

	void Visit(const DeclarationStatement *statement) override
	{
		statement->GetDeclaration()->Accept(*this);
	}

	void Visit(const InstructionStatement *statement) override
	{
		for (const auto& operand : statement->GetOperands())
		{
			operand->Accept(*this);
		}
	}

	void Visit(const LabelStatement *statement) override
	{
		statement->GetLabel()->Accept(*this);
	}

	// Formatting

	void PrintResults(const FunctionDefinition<VoidType> *function) const
	{
		Utils::Logger::LogInfo(m_name + " '" + function->GetName() + "'");

		std::stringstream string;
		m_analysisSet.Print(string, 1);

		Utils::Logger::LogInfo(string.str(), 0, true, Utils::Logger::NoPrefix);
	}

	// Accessors

	const F& GetAnalysisSet() const { return m_analysisSet; }

	// Options

	const std::string& GetName() const { return m_name; }
	const std::string& GetShortName() const { return m_shortName; }

protected:
	F m_analysisSet;

	// Naming

	const std::string m_name;
	const std::string m_shortName;
};

}
}
