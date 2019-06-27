#include "HorseIR/Semantics/DefinitelyAssigned.h"

#include <algorithm>

#include "HorseIR/Tree/Tree.h"
#include "HorseIR/Utils/PrettyPrinter.h"

#include "Utils/Logger.h"

namespace HorseIR {

void DefinitelyAssigned::Analyze(const Program *program)
{
	// Traverse the program in sections, first collecting the globals, and then handling
	// the function bodies

	m_globalsPhase = true;
	program->Accept(*this);

	m_globalsPhase = false;
	program->Accept(*this);
}

bool DefinitelyAssigned::VisitIn(const GlobalDeclaration *global)
{
	// If in globals mode, add the definition to the set

	if (m_globalsPhase)
	{
		m_globals.insert(global->GetDeclaration()->GetSymbol());
	}
	return false;
}

bool DefinitelyAssigned::VisitIn(const Function *function)
{
	// If we are in the globals mode, skip the function. Otherwise, handle the body with
	// the globals definitions

	if (!m_globalsPhase)
	{
		m_definitions = m_globals;
		return true;
	}
	return false;
}

bool DefinitelyAssigned::VisitIn(const Parameter *parameter)
{
	// We assume all parameters are defined

	m_definitions.insert(parameter->GetSymbol());
	return true;
}

bool DefinitelyAssigned::VisitIn(const AssignStatement *assignS)
{
	// Check the RHS expression, then add the definitions to the set

	assignS->GetExpression()->Accept(*this);
	for (const auto& target : assignS->GetTargets())
	{
		m_definitions.insert(target->GetSymbol());
	}
	return false;
}

bool DefinitelyAssigned::VisitIn(const IfStatement *ifS)
{
	// Traverse the condition before the branches

	ifS->GetCondition()->Accept(*this);

	// Compute the set for the true branch

	const auto inSet = m_definitions;
	ifS->GetTrueBlock()->Accept(*this);
	const auto trueSet = m_definitions;

	if (ifS->HasElseBranch())
	{
		// Compute the set for the else branch

		m_definitions = inSet;
		ifS->GetElseBlock()->Accept(*this);
		const auto elseSet = m_definitions;

		// Intersect the sets from both branches

		m_definitions.clear();
		for (const auto& definition : trueSet)
		{
			if (elseSet.find(definition) != elseSet.end())
			{
				m_definitions.insert(definition);
			}
		}
	}
	else
	{
		// If there is no else branch, we ignore all definitions from "true"

		m_definitions = inSet;
	}

	return false;
}

bool DefinitelyAssigned::VisitIn(const WhileStatement *whileS)
{
	// Backup/restore definitions before/after the body. Definitions are discarded
	// as the loop may be skipped entirely

	const auto inSet = m_definitions;
	whileS->GetCondition()->Accept(*this);
	whileS->GetBody()->Accept(*this);
	m_definitions = inSet;

	return false;
}

bool DefinitelyAssigned::VisitIn(const RepeatStatement *repeatS)
{
	// Backup/restore definitions before/after the body. Definitions are discarded
	// as the loop may be skipped entirely

	const auto inSet = m_definitions;
	repeatS->GetCondition()->Accept(*this);
	repeatS->GetBody()->Accept(*this);
	m_definitions = inSet;

	return false;
}

bool DefinitelyAssigned::VisitIn(const FunctionLiteral *literal)
{
	// Do not traverse the function identifier

	return false;
}

bool DefinitelyAssigned::VisitIn(const Identifier *identifier)
{
	// Check that the variable is defined on all paths (anything in the set has this property)

	if (m_definitions.find(identifier->GetSymbol()) == m_definitions.end())
	{
		Utils::Logger::LogError("Variable '" + PrettyPrinter::PrettyString(dynamic_cast<const Operand *>(identifier)) + "' is not defined on all paths.");
	}
	return true;
}

}
