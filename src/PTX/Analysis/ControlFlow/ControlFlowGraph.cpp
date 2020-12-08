#include "PTX/Analysis/ControlFlow/ControlFlowGraph.h"

#include <regex>
#include <unordered_map>

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
		auto label = std::regex_replace(node->ToDOTString(), std::regex("\""), "\\\"");

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

			string += "\tn_" + nodeIndex + " -> n_" + successorIndex;
			string += " [ltail=cluster_" + nodeIndex + ",lhead=cluster_" + successorIndex + ",label=\" \"];\n";
		}
	}

	// Complete with label for function name

	string += "\tlabel=\"" + m_function->GetName() + "\"\n";
	string += "}";

	return string;
}
 
}
}
