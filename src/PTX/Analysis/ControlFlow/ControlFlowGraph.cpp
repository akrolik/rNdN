#include "PTX/Analysis/ControlFlow/ControlFlowGraph.h"

#include <regex>
#include <unordered_map>

#include "PTX/Tree/Tree.h"
#include "PTX/Utils/PrettyPrinter.h"

namespace PTX {
namespace Analysis {

std::string ControlFlowGraph::ToDOTString() const
{
	std::unordered_map<ControlFlowNode, unsigned int> indexMap;
	auto index = 1u;

 	// Construct the DOT surrounding structure

	auto string = "digraph CFG_" + m_function->GetName() + " {\n";
	string += "\tcompound=true;\n";

	// Add all CFG nodes (basic blocks) in the graph

	for (const auto& node : GetNodes())
	{
		indexMap[node] = index;
		auto name = "n_" + std::to_string(index);

		const auto& statements = node->GetStatements();
		std::string statementString;
		if (statements.size() == 0)
		{
			statementString = "%empty%";
		}
		else if (statements.size() <= 3)
		{
			for (const auto& statement : statements)
			{
				statementString += PrettyPrinter::PrettyString(statement, true) + "\\l";
			}
		}
		else
		{
			statementString += PrettyPrinter::PrettyString(statements.at(0), true) + "\\l";
			statementString += PrettyPrinter::PrettyString(statements.at(1), true) + "\\l";
			statementString += "[...]\\l";
			statementString += PrettyPrinter::PrettyString(statements.back(), true) + "\\l";
		}

		auto label = std::regex_replace(statementString, std::regex("\""), "\\\"");

		string += "\tsubgraph cluster_" + std::to_string(index) + "{\n";
		string += "\t\t" + name + "[label=\"" + label + "\", shape=plaintext]\n"; 
		string += "\t\tlabel=\"" + node->GetLabel()->GetName() + "\"\n";
		string += "\t}\n";

		index++;
	}
	string +"\n";

	// Add all edges in the graph

	for (const auto& node : m_nodes)
	{
		for (const auto& successor : GetSuccessors(node))
		{
			auto nodeIndex = std::to_string(indexMap[node]);
			auto successorIndex = std::to_string(indexMap[successor]);

			std::string label = " ";
			if (auto [predicate, negate] = GetEdgeData(node, successor); predicate != nullptr)
			{
				label = ((negate) ? "!" : "") + predicate->ToString();
			}

			string += "\tn_" + nodeIndex + " -> n_" + successorIndex;
			string += " [ltail=cluster_" + nodeIndex + ",lhead=cluster_" + successorIndex + ",label=\"" + label + "\"];\n";
		}
	}

	// Complete with label for function name

	string += "\tlabel=\"" + m_function->GetName() + "\"\n";
	string += "}";

	return string;
}
 
}
}
