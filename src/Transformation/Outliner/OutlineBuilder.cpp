#include "Transformation/Outliner/OutlineBuilder.h"

#include <queue>

namespace Transformation {

void OutlineBuilder::Build(const Analysis::CompatibilityOverlay *overlay)
{
	overlay->Accept(*this);
}

void OutlineBuilder::Visit(const Analysis::CompatibilityOverlay *overlay)
{
	auto graph = overlay->GetGraph();

	// Queue: store the current nodes 0 in-degree
	// Edges: count the in-degree of each node

	std::queue<const HorseIR::Statement *> m_queue;
	std::unordered_map<const HorseIR::Statement *, unsigned int> m_edges;

	// Initialization with root nodes and count for incoming edges of each node

	for (const auto& statement : overlay->GetStatements())
	{
		auto count = overlay->GetGraph()->GetOutDegree(statement);
		if (count == 0)
		{
			m_queue.push(statement);
		}
		m_edges.insert({statement, count});
	}

	//TODO: Technically, there is no need to topological sort, because we should already have an order that works from the partition, unless the partition is only graph based with sets

	// Perform the topological sort

	while (!m_queue.empty())
	{
		//TODO: Avoid const_cast, need to deep copy the AST probably

		auto statement = const_cast<HorseIR::Statement *>(m_queue.front());
		m_queue.pop();

		for (auto& destination : graph->GetIncomingEdges(statement))
		{
			m_edges[destination]--;
			if (m_edges[destination] == 0)
			{
				m_queue.push(destination);
			}
		}

		// Insert the statement into the new function

		auto& statements = m_statements.top();
		//TODO: Check if there is a function already
		statements.insert(std::begin(statements), statement);
	}
}

void OutlineBuilder::Visit(const Analysis::FunctionCompatibilityOverlay *overlay)
{
	// Add the original container function if it exists

	auto containerFunction = overlay->GetNode();
	if (containerFunction != nullptr)
	{
		m_containerFunctions.push({containerFunction, 0});
	}

	// Traversing the function body and accumulate statements

	m_statements.emplace();
	CompatibilityOverlayConstVisitor::Visit(overlay);

	// Create a new function with the entry function and body

	auto& [currentFunction, currentIndex] = m_containerFunctions.top();
	auto name = currentFunction->GetName();
	auto index = currentIndex++;
	if (index > 0)
	{
       		name += "_" + std::to_string(index);
	}

	auto& parameters = currentFunction->GetParameters();
	auto& returnTypes = currentFunction->GetReturnTypes();
	auto kernel = true;

	m_functions.insert(std::begin(m_functions), new HorseIR::Function(name, parameters, returnTypes, m_statements.top(), kernel));

	// Clear the context as needed

	if (containerFunction != nullptr)
	{
		m_containerFunctions.pop();
	}
	m_statements.pop();
}

void OutlineBuilder::Visit(const Analysis::IfCompatibilityOverlay *overlay)
{
	// Construct the true block of statements from the child overlay

	m_statements.emplace();
	//TODO: Build
	// overlay->GetTrueOverlay()->Accept(*this);

	auto trueBlock = new HorseIR::BlockStatement(m_statements.top());
	m_statements.pop();

	// Construct the true block of statements from the child overlay

	m_statements.emplace();
	//TODO: Build
	// overlay->GetElseOverlay()->Accept(*this);

	auto elseBlock = new HorseIR::BlockStatement(m_statements.top());
	m_statements.pop();

	// Create if statement with condition and both blocks

	auto condition = overlay->GetNode()->GetCondition();
	auto statement = new HorseIR::IfStatement(condition, trueBlock, elseBlock);

	// Insert into the current statement list

	auto& statements = m_statements.top();
	statements.insert(std::begin(statements), statement);
}

void OutlineBuilder::Visit(const Analysis::WhileCompatibilityOverlay *overlay)
{
	// Construct the body from the child overlay

	m_statements.emplace();
	//TODO: Build
	// overlay->GetBodyOverlay()->Accept(*this);

	auto block = new HorseIR::BlockStatement(m_statements.top());
	m_statements.pop();

	// Create while statement with condition and body

	auto condition = overlay->GetNode()->GetCondition();
	auto statement = new HorseIR::WhileStatement(condition, block);

	// Insert into the current statement list

	auto& statements = m_statements.top();
	statements.insert(std::begin(statements), statement);
}

void OutlineBuilder::Visit(const Analysis::RepeatCompatibilityOverlay *overlay)
{
	// Construct the body from the child overlay

	m_statements.emplace();
	//TODO: Build
	// overlay->GetBodyOverlay()->Accept(*this);

	auto block = new HorseIR::BlockStatement(m_statements.top());
	m_statements.pop();

	// Create repeat statement with condition and body

	auto condition = overlay->GetNode()->GetCondition();
	auto statement = new HorseIR::RepeatStatement(condition, block);

	// Insert into the current statement list

	auto& statements = m_statements.top();
	statements.insert(std::begin(statements), statement);
}

}
