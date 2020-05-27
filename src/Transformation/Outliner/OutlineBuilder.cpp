#include "Transformation/Outliner/OutlineBuilder.h"

#include "Analysis/Dependency/Overlay/DependencyOverlay.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Variant.h"

namespace Transformation {

void OutlineBuilder::Build(const Analysis::FunctionDependencyOverlay *overlay)
{
	auto timeBuilder_start = Utils::Chrono::Start("Outline builder '" + std::string(overlay->GetName()) + "'");
	overlay->Accept(*this);
	Utils::Chrono::End(timeBuilder_start);
}

void OutlineBuilder::Visit(const HorseIR::Statement *statement)
{
	// Regular statements are clone before being inserted

	InsertStatement(statement->Clone());
}

void OutlineBuilder::Visit(const HorseIR::AssignStatement *assignS)
{
	// Transform the assignment statement, removing all declarations. The declarations
	// are accumulated and inserted at the top of the function/kernel

	std::vector<HorseIR::LValue *> targets;
	for (const auto& target : assignS->GetTargets())
	{
		auto symbol = target->GetSymbol();
		m_symbols.top().insert(symbol);
		targets.push_back(new HorseIR::Identifier(symbol->name));
	}

	auto expression = assignS->GetExpression();
	InsertStatement(new HorseIR::AssignStatement(targets, expression->Clone()));
}

void OutlineBuilder::InsertStatement(HorseIR::Statement *statement)
{
	m_statements.top().push_back(statement);
}

void OutlineBuilder::InsertDeclaration(HorseIR::DeclarationStatement *declaration)
{
	auto& statements = m_statements.top();
	statements.insert(std::begin(statements), declaration);
}

const HorseIR::Type *OutlineBuilder::GetType(const HorseIR::SymbolTable::Symbol *symbol)
{
	if (symbol->kind == HorseIR::SymbolTable::Symbol::Kind::Variable)
	{
		return dynamic_cast<const HorseIR::VariableDeclaration *>(symbol->node)->GetType();
	}

	Utils::Logger::LogError("'" + symbol->name + "' does not name a variable");
}

void OutlineBuilder::BuildDeclarations()
{
	// For each symbol in the current set of symbols, add a declaration to the top of the current function

	for (const auto& symbol : m_symbols.top())
	{
		auto type = GetType(symbol);
		InsertDeclaration(new HorseIR::DeclarationStatement(
			new HorseIR::VariableDeclaration(symbol->name, type->Clone())
		));
	}
}

void OutlineBuilder::Visit(const Analysis::DependencyOverlay *overlay)
{
	// Construct a new kernel if needed

	bool isKernel = (overlay->IsGPU() && !overlay->GetParent()->IsGPU());
	if (isKernel)
	{
		m_statements.emplace();
	}

	// Collect the declarations for this block

	m_symbols.emplace();

	// Perform the topological sort and construct the statement list recursively

	const auto subgraph = overlay->GetSubgraph();
	subgraph->TopologicalOrdering([&](Analysis::DependencySubgraph::OrderingContext& context, const Analysis::DependencySubgraphNode& node)
	{
		std::visit(overloaded
		{
			[&](const HorseIR::Statement *statement)
			{
				// Insert the statement into the statement list

				if (subgraph->IsGPULibraryNode(statement))
				{
					InsertStatement(m_libraryOutliner.Outline(statement));
				}
				else
				{
					statement->Accept(*this);
				}
			},
			[&](const Analysis::DependencyOverlay *childOverlay)
			{
				childOverlay->Accept(*this);
			}},
			node
		);
		return true;
	});

	// Collect the incoming and outgoing symbols

	std::unordered_set<const HorseIR::SymbolTable::Symbol *> inSet;
	std::unordered_set<const HorseIR::SymbolTable::Symbol *> outSet;

	const auto containerGraph = overlay->GetParent()->GetSubgraph();
	for (const auto& predecessor : containerGraph->GetPredecessors(overlay))
	{
		const auto& symbols = containerGraph->GetEdgeData(predecessor, overlay);
		inSet.insert(std::begin(symbols), std::end(symbols));
	}

	for (const auto& successor : containerGraph->GetSuccessors(overlay))
	{
		const auto& symbols = containerGraph->GetEdgeData(overlay, successor);
		outSet.insert(std::begin(symbols), std::end(symbols));
	}

	// Remove incoming symbols from the local declaration set (handles outside declarations)

	for (auto symbol : inSet)
	{
		m_symbols.top().erase(symbol);
	}

	// Add all variable declarations to the top of the block

	BuildDeclarations();
	m_symbols.pop();

	if (isKernel)
	{
		// Collect the input paramters (name+type) and return types

		std::vector<const HorseIR::SymbolTable::Symbol *> inSymbols(std::begin(inSet), std::end(inSet));
		std::vector<const HorseIR::SymbolTable::Symbol *> outSymbols(std::begin(outSet), std::end(outSet));

		std::vector<HorseIR::Parameter *> parameters;
		std::vector<HorseIR::Type *> returnTypes;

		for (const auto& symbol : inSymbols)
		{
			auto type = GetType(symbol);
			parameters.push_back(new HorseIR::Parameter(symbol->name, type->Clone()));
		}

		for (const auto& symbol : outSymbols)
		{
			auto type = GetType(symbol);
			returnTypes.push_back(type->Clone());
		}

		// Create a return statement at the end of the kernel

		std::vector<HorseIR::Operand *> returnOperands;
		for (const auto& symbol : outSymbols)
		{
			returnOperands.push_back(new HorseIR::Identifier(symbol->name));
		}
		m_statements.top().push_back(new HorseIR::ReturnStatement(returnOperands));

		// Construct the new kernel function

		auto name = m_currentFunction->GetName() + "_" + std::to_string(m_kernelIndex++);
		auto kernel = new HorseIR::Function(
			name,
			parameters,
			returnTypes,
			m_statements.top(),
			true
		);

		m_functions.push_back(kernel); 
		m_statements.pop();

		// Insert a call to the new kernel in the containing function

		std::vector<HorseIR::Operand *> callOperands;
		for (const auto& symbol : inSymbols)
		{
			callOperands.push_back(new HorseIR::Identifier(symbol->name));
		}
		auto call = new HorseIR::CallExpression(
			new HorseIR::FunctionLiteral(new HorseIR::Identifier(name)),
			callOperands
		);

		// Generate the containing statement and add the output symbols to the outer scope

		m_symbols.top().insert(std::begin(outSymbols), std::end(outSymbols));

		if (outSymbols.size() > 0)
		{
			// Create the assignment statement (at least one return value exists)

			std::vector<HorseIR::LValue *> assignLValues;
			for (const auto& symbol : outSymbols)
			{
				assignLValues.push_back(new HorseIR::Identifier(symbol->name));
			}
			InsertStatement(new HorseIR::AssignStatement(assignLValues, call));
		}
		else
		{
			// Create an expression statement (no return values)

			InsertStatement(new HorseIR::ExpressionStatement(call));
		}
	}
}

void OutlineBuilder::Visit(const Analysis::FunctionDependencyOverlay *overlay)
{
	// Traversing the function body and accumulate statements

	m_currentFunction = overlay->GetNode();

	m_statements.emplace();
	m_symbols.emplace();

	overlay->GetBody()->Accept(*this);

	// Create a new function with the entry function and body, cloning inputs/outputs

	std::vector<HorseIR::Parameter *> parameters;
	for (const auto& parameter : m_currentFunction->GetParameters())
	{
		parameters.push_back(parameter->Clone());
	}

	std::vector<HorseIR::Type *> returnTypes;
	for (const auto& returnType : m_currentFunction->GetReturnTypes())
	{
		returnTypes.push_back(returnType->Clone());
	}

	// Add declaration statements to the top of the function

	BuildDeclarations();

	auto newFunction = new HorseIR::Function(
		m_currentFunction->GetName(),
		parameters,
		returnTypes,
		m_statements.top(),
		false
	);

	m_functions.insert(std::begin(m_functions), newFunction); 

	m_statements.pop();
	m_symbols.pop();

	m_currentFunction = nullptr;
}

void OutlineBuilder::Visit(const Analysis::IfDependencyOverlay *overlay)
{
	// Construct the true block of statements from the child overlay

	m_statements.emplace();
	overlay->GetTrueBranch()->Accept(*this);

	auto trueBlock = new HorseIR::BlockStatement(m_statements.top());
	m_statements.pop();

	// Construct the else block of statements from the child overlay

	HorseIR::BlockStatement *elseBlock = nullptr;
	if (overlay->HasElseBranch())
	{
		m_statements.emplace();
		overlay->GetElseBranch()->Accept(*this);

		elseBlock = new HorseIR::BlockStatement(m_statements.top());
		m_statements.pop();
	}

	// Create if statement with condition and both blocks

	auto condition = overlay->GetNode()->GetCondition();
	InsertStatement(new HorseIR::IfStatement(condition->Clone(), trueBlock, elseBlock));
}

void OutlineBuilder::Visit(const Analysis::WhileDependencyOverlay *overlay)
{
	// Construct the body from the child overlay

	m_statements.emplace();
	overlay->GetBody()->Accept(*this);

	auto block = new HorseIR::BlockStatement(m_statements.top());
	m_statements.pop();

	// Create while statement with condition and body

	auto condition = overlay->GetNode()->GetCondition();
	InsertStatement(new HorseIR::WhileStatement(condition->Clone(), block));
}

void OutlineBuilder::Visit(const Analysis::RepeatDependencyOverlay *overlay)
{
	// Construct the body from the child overlay

	m_statements.emplace();
	overlay->GetBody()->Accept(*this);

	auto block = new HorseIR::BlockStatement(m_statements.top());
	m_statements.pop();

	// Create repeat statement with condition and body

	auto condition = overlay->GetNode()->GetCondition();
	InsertStatement(new HorseIR::RepeatStatement(condition->Clone(), block));
}

}
