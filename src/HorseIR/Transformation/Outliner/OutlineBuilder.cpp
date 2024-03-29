#include "HorseIR/Transformation/Outliner/OutlineBuilder.h"

#include "HorseIR/Analysis/Dependency/DependencyGraph.h"
#include "HorseIR/Analysis/Dependency/Overlay/DependencyOverlay.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Variant.h"

namespace HorseIR {
namespace Transformation {

const std::vector<Function *>& OutlineBuilder::Build(const Analysis::FunctionDependencyOverlay *overlay)
{
	auto timeBuilder_start = Utils::Chrono::Start("Outline builder '" + std::string(overlay->GetName()) + "'");
	overlay->Accept(*this);
	Utils::Chrono::End(timeBuilder_start);

	return m_functions;
}

void OutlineBuilder::Visit(const Statement *statement)
{
	// Regular statements are clone before being inserted

	InsertStatement(statement->Clone());
}

void OutlineBuilder::Visit(const AssignStatement *assignS)
{
	// Transform the assignment statement, removing all declarations. The declarations
	// are accumulated and inserted at the top of the function/kernel

	std::vector<LValue *> targets;
	for (const auto& target : assignS->GetTargets())
	{
		auto symbol = target->GetSymbol();
		m_symbols.top().insert(symbol);
		targets.push_back(new Identifier(symbol->name));
	}

	auto expression = assignS->GetExpression();
	InsertStatement(new AssignStatement(targets, expression->Clone()));
}

void OutlineBuilder::InsertStatement(Statement *statement)
{
	auto& statements = m_statements.top();
	statements.push_back(statement);
}

void OutlineBuilder::InsertDeclaration(DeclarationStatement *declaration)
{
	auto& statements = m_statements.top();
	statements.insert(std::begin(statements), declaration);
}

const Type *OutlineBuilder::GetType(const SymbolTable::Symbol *symbol)
{
	if (symbol->kind == SymbolTable::Symbol::Kind::Variable)
	{
		return dynamic_cast<const VariableDeclaration *>(symbol->node)->GetType();
	}

	Utils::Logger::LogError("'" + symbol->name + "' does not name a variable");
}

void OutlineBuilder::BuildDeclarations()
{
	// For each symbol in the current set of symbols, add a declaration to the top of the current function

	auto& top = m_symbols.top();
	std::vector<const SymbolTable::Symbol *> symbols(std::begin(top), std::end(top));

	std::sort(symbols.begin(), symbols.end(), [](const SymbolTable::Symbol *s1, const SymbolTable::Symbol *s2)
	{
		return s1->name > s2->name;
	});

	for (const auto& symbol : symbols)
	{
		auto type = GetType(symbol);
		InsertDeclaration(new DeclarationStatement(
			new VariableDeclaration(symbol->name, type->Clone())
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
	subgraph->TopologicalOrderDFS([&](Analysis::DependencySubgraph::OrderContextDFS& context, const Analysis::DependencySubgraphNode& node)
	{
		std::visit(overloaded
		{
			[&](const Statement *statement)
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

	robin_hood::unordered_set<const SymbolTable::Symbol *> inSet;
	robin_hood::unordered_set<const SymbolTable::Symbol *> outSet;

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

		std::vector<const SymbolTable::Symbol *> inSymbols(std::begin(inSet), std::end(inSet));
		std::vector<const SymbolTable::Symbol *> outSymbols(std::begin(outSet), std::end(outSet));

		// Order by name for determinism

		auto sortFunction = [](const SymbolTable::Symbol *s1, const SymbolTable::Symbol *s2)
		{
			return s1->name < s2->name;
		};

		std::sort(inSymbols.begin(), inSymbols.end(), sortFunction);
		std::sort(outSymbols.begin(), outSymbols.end(), sortFunction);

		// Construct parameters/return types for function signature

		std::vector<Parameter *> parameters;
		std::vector<Type *> returnTypes;

		for (const auto& symbol : inSymbols)
		{
			auto type = GetType(symbol);
			parameters.push_back(new Parameter(symbol->name, type->Clone()));
		}

		for (const auto& symbol : outSymbols)
		{
			auto type = GetType(symbol);
			returnTypes.push_back(type->Clone());
		}

		// Create a return statement at the end of the kernel

		std::vector<Operand *> returnOperands;
		for (const auto& symbol : outSymbols)
		{
			returnOperands.push_back(new Identifier(symbol->name));
		}
		m_statements.top().push_back(new ReturnStatement(returnOperands));

		// Construct the new kernel function

		auto name = m_currentFunction->GetName() + "_" + std::to_string(m_kernelIndex++);
		auto kernel = new Function(
			name,
			parameters,
			returnTypes,
			m_statements.top(),
			true
		);

		m_functions.push_back(kernel); 
		m_statements.pop();

		// Insert a call to the new kernel in the containing function

		std::vector<Operand *> callOperands;
		for (const auto& symbol : inSymbols)
		{
			callOperands.push_back(new Identifier(symbol->name));
		}
		auto call = new CallExpression(
			new FunctionLiteral(new Identifier(name)),
			callOperands
		);

		// Generate the containing statement and add the output symbols to the outer scope

		m_symbols.top().insert(std::begin(outSymbols), std::end(outSymbols));

		if (outSymbols.size() > 0)
		{
			// Create the assignment statement (at least one return value exists)

			std::vector<LValue *> assignLValues;
			for (const auto& symbol : outSymbols)
			{
				assignLValues.push_back(new Identifier(symbol->name));
			}
			InsertStatement(new AssignStatement(assignLValues, call));
		}
		else
		{
			// Create an expression statement (no return values)

			InsertStatement(new ExpressionStatement(call));
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

	std::vector<Parameter *> parameters;
	for (const auto& parameter : m_currentFunction->GetParameters())
	{
		parameters.push_back(parameter->Clone());
	}

	std::vector<Type *> returnTypes;
	for (const auto& returnType : m_currentFunction->GetReturnTypes())
	{
		returnTypes.push_back(returnType->Clone());
	}

	// Add declaration statements to the top of the function

	BuildDeclarations();

	auto newFunction = new Function(
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

	auto trueBlock = new BlockStatement(m_statements.top());
	m_statements.pop();

	// Construct the else block of statements from the child overlay

	BlockStatement *elseBlock = nullptr;
	if (overlay->HasElseBranch())
	{
		m_statements.emplace();
		overlay->GetElseBranch()->Accept(*this);

		elseBlock = new BlockStatement(m_statements.top());
		m_statements.pop();
	}

	// Create if statement with condition and both blocks

	auto condition = overlay->GetNode()->GetCondition();
	InsertStatement(new IfStatement(condition->Clone(), trueBlock, elseBlock));
}

void OutlineBuilder::Visit(const Analysis::WhileDependencyOverlay *overlay)
{
	// Construct the body from the child overlay

	m_statements.emplace();
	overlay->GetBody()->Accept(*this);

	auto block = new BlockStatement(m_statements.top());
	m_statements.pop();

	// Create while statement with condition and body

	auto condition = overlay->GetNode()->GetCondition();
	InsertStatement(new WhileStatement(condition->Clone(), block));
}

void OutlineBuilder::Visit(const Analysis::RepeatDependencyOverlay *overlay)
{
	// Construct the body from the child overlay

	m_statements.emplace();
	overlay->GetBody()->Accept(*this);

	auto block = new BlockStatement(m_statements.top());
	m_statements.pop();

	// Create repeat statement with condition and body

	auto condition = overlay->GetNode()->GetCondition();
	InsertStatement(new RepeatStatement(condition->Clone(), block));
}

}
}
