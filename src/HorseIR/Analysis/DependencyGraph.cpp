#include "HorseIR/Analysis/DependencyGraph.h"

#include "HorseIR/Tree/Method.h"

namespace HorseIR {

void DependencyGraph::InsertDeclaration(const Declaration *declaration)
{
	m_declarations.insert(declaration);
}

void DependencyGraph::InsertStatement(const Statement *statement)
{
	m_statements.insert(statement);
}

void DependencyGraph::InsertDefinition(const Declaration *declaration, const Statement *statement)
{
	m_definitions[declaration] = statement;
}

void DependencyGraph::InsertDependency(const Statement *statement, const Declaration *declaration)
{
	m_outgoingEdges[declaration].insert(statement);
	m_incomingEdges[statement].insert(declaration);
}

std::string DependencyGraph::ToString() const
{
	std::string code;
	auto name = m_method->GetName();
	code += "\tsubgraph cluster_" + name + " {\n";
	
	unsigned int i = 0;
	std::unordered_map<const Node *, std::string> nameMap;

	for (const auto& node : m_declarations)
	{
		auto nodeNode = name + std::to_string(i);
		if (m_definitions.find(node) == m_definitions.end())
		{
			code += "\t\t" + nodeNode + "[label=\"" + node->ToString() + "\"];\n";
			nameMap[node] = nodeNode;
			++i;
		}
	}

	for (const auto& node : m_statements)
	{
		auto nodeNode = name + std::to_string(i);
		code += "\t\t" + nodeNode + "[label=\"" + node->ToString() + "\", shape=rectangle];\n";
		nameMap[node] = nodeNode;
		++i;
	}

	for (const auto& entry : m_outgoingEdges)
	{
		auto parent = entry.first;
		if (m_definitions.find(parent) == m_definitions.end())
		{
			for (const auto& child : entry.second)
			{
				code += "\t\t" + nameMap[parent] + " -> " + nameMap[child] + ";\n";
			}
		}
		else
		{
			for (const auto& child : entry.second)
			{
				code += "\t\t" + nameMap[m_definitions.at(parent)] + " -> " + nameMap[child] + ";\n";
			}
		}
	}

	code += "\t\tlabel=\"" + m_method->SignatureString() + "\";\n";
	code += "\t}";
	return code;
}

void GlobalDependencyGraph::InsertDependencies(const Method *method, DependencyGraph *dependencies)
{
	m_methodDependencies.insert({method, dependencies});
}

std::string GlobalDependencyGraph::ToString() const
{
	std::string code;
	code += "digraph dependencies {\n";

	for (const auto& entry : m_methodDependencies)
	{
		code += entry.second->ToString() + "\n";
	}

	code += "}";
	return code;
}

}
