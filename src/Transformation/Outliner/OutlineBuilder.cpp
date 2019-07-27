#include "Transformation/Outliner/OutlineBuilder.h"

#include "Utils/Variant.h"

#include <queue>

namespace Transformation {

void OutlineBuilder::Build(const Analysis::CompatibilityOverlay *overlay)
{
	overlay->Accept(*this);
}

Analysis::CompatibilityOverlay *OutlineBuilder::GetChildOverlay(const std::vector<Analysis::CompatibilityOverlay *>& childOverlays, const HorseIR::Statement *statement) const
{
	// Return the top-level child overlay which contains the statement

	for (const auto child : childOverlays)
	{
		// Check the current overlay for the statement

		if (child->ContainsStatement(statement))
		{
			return child;
		}
	}

	// Check all children overlays, note that we want the highest *containing* overlay, not the actual overlay

	for (const auto child : childOverlays)
	{
		if (GetChildOverlay(child->GetChildren(), statement) != nullptr)
		{
			return child;
		}
	}

	// Else, this is contained in a sibling or parent overlay

	return nullptr;
}


unsigned int OutlineBuilder::GetOutDegree(const Analysis::CompatibilityOverlay *overlay) const
{
	// Count the number of outgoing edges with destinations either in the parent or sibling overlay

	// If there is no parent, then all edges are internal to the overlay

	auto parent = overlay->GetParent();
	if (parent == nullptr)
	{
		return 0;
	}

	auto graph = overlay->GetGraph();
	auto edges = 0u;
	
	for (const auto statement : overlay->GetStatements())
	{
		for (const auto destination : graph->GetOutgoingEdges(statement))
		{
			if (graph->IsBackEdge(statement, destination))
			{
				continue;
			}

			// Check if the destination is in the parent overlay

			if (!parent->ContainsStatement(destination))
			{
				// Check if the destination is in a sibling overlay

				//TODO: Think about back edges
				auto destinationOverlay = GetChildOverlay(parent->GetChildren(), destination);
				if (destinationOverlay == nullptr || destinationOverlay == overlay)
				{
					continue;
				}
			}
			edges++;
		}
	}
	
	return edges;
}

void OutlineBuilder::Visit(const Analysis::CompatibilityOverlay *overlay)
{
	auto graph = overlay->GetGraph();

	// Queue: store the current nodes 0 in-degree
	// Edges: count the in-degree of each node

	using NodeType = std::variant<const HorseIR::Statement *, const Analysis::CompatibilityOverlay *>;

	std::queue<NodeType> queue;
	std::unordered_map<NodeType, unsigned int> edges;

	// Initialization with root nodes and count for incoming edges of each node

	for (const auto& statement : overlay->GetStatements())
	{
		auto count = 0u;
		for (const auto destination : graph->GetOutgoingEdges(statement))
		{
			// Check if the destination is in the parent overlay

			if (graph->IsBackEdge(statement, destination))
			{
				continue;
			}

			if (!overlay->ContainsStatement(destination))
			{
				// Check if the destination is in a sibling overlay

				auto destinationOverlay = GetChildOverlay(overlay->GetChildren(), destination);
				if (destinationOverlay == nullptr)
				{
					continue;
				}
			}
			count++;
		}

		if (count == 0)
		{
			queue.push(statement);
		}
		edges.insert({statement, count});
	}

	for (const auto& child : overlay->GetChildren())
	{
		auto count = GetOutDegree(child);
		if (count == 0)
		{
			queue.push(child);
		}
		edges.insert({child, count});
	}

	// Perform the topological sort

	while (!queue.empty())
	{
		const auto& node = queue.front();
		queue.pop();

		std::visit(overloaded {

			[&](const HorseIR::Statement *statement)
			{
				for (auto& destination : graph->GetIncomingEdges(statement))
				{
					if (overlay->ContainsStatement(destination))
					{
						// Decrease the degree of the destination if it's in the current overlay

						edges[destination]--;
						if (edges[destination] == 0)
						{
							queue.push(destination);
						}
					}
					else
					{
						// Check for a child overlay destination to decrease

						auto destinationOverlay = GetChildOverlay(overlay->GetChildren(), destination);
						if (destinationOverlay != nullptr)
						{
							edges[destinationOverlay]--;
							if (edges[destinationOverlay] == 0)
							{
								queue.push(destinationOverlay);
							}
						}
					}
				}

				// Insert the statement into the statement list
				//TODO: Avoid const_cast, need to deep copy the AST probably

				auto& statements = m_statements.top();
				statements.insert(std::begin(statements), const_cast<HorseIR::Statement *>(statement));
			},
			[&](const Analysis::CompatibilityOverlay *childOverlay)
			{
				//TODO: Check that we haven't double nested the kernel
				if (childOverlay->IsGPU())
				{
					// Traversing the kernel body and accumulate statements

					m_statements.emplace();
					childOverlay->Accept(*this);

					auto index = m_kernelIndex++;
					auto name = m_currentFunction->GetName() + "_" + std::to_string(index);

					auto kernel = new HorseIR::Function(
						name,
						{},
						{},
						m_statements.top(),
						true
					);

					m_functions.push_back(kernel); 
					m_statements.pop();

					//TODO: We need to insert a call in the previous statement group with the incoming stuff
				}
				else
				{
					childOverlay->Accept(*this);
				}

				//TODO: Decrease all other incoming edges
			}},
			node
		);
	}
}

void OutlineBuilder::Visit(const Analysis::FunctionCompatibilityOverlay *overlay)
{
	// Traversing the function body and accumulate statements

	auto oldFunction = overlay->GetNode();
	m_currentFunction = oldFunction;

	m_statements.emplace();
	overlay->GetBody()->Accept(*this);

	// Create a new function with the entry function and body

	auto newFunction = new HorseIR::Function(
		m_currentFunction->GetName(),
		m_currentFunction->GetParameters(),
		m_currentFunction->GetReturnTypes(),
		m_statements.top(),
		false
	);

	m_functions.insert(std::begin(m_functions), newFunction); 
	m_statements.pop();
}

void OutlineBuilder::Visit(const Analysis::IfCompatibilityOverlay *overlay)
{
	// Construct the true block of statements from the child overlay

	m_statements.emplace();
	overlay->GetTrueBranch()->Accept(*this);

	auto trueBlock = new HorseIR::BlockStatement(m_statements.top());
	m_statements.pop();

	// Construct the true block of statements from the child overlay

	m_statements.emplace();
	overlay->GetElseBranch()->Accept(*this);

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
	overlay->GetBody()->Accept(*this);

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
	overlay->GetBody()->Accept(*this);

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
