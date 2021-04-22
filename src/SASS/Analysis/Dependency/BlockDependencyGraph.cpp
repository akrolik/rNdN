#include "SASS/Analysis/Dependency/BlockDependencyGraph.h"

#include "Libraries/robin_hood.h"

namespace SASS {
namespace Analysis {

std::string BlockDependencyGraph::ToDOTString() const
{
	robin_hood::unordered_map<Instruction *, unsigned int> indexMap;
	auto index = 1u;

 	// Construct the DOT surrounding structure

	auto string = "digraph dependency_" + m_block->GetName() + " {\n";

	// Add all CFG nodes (basic blocks) in the graph

	for (const auto& node : GetNodes())
	{
		indexMap[node] = index;

		auto name = "n_" + std::to_string(index);
		auto label = node->ToString();

		string += "\t" + name + "[label=\"" + label + "\", shape=plaintext]\n"; 

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

			string += "\tn_" + nodeIndex + " -> n_" + successorIndex + ";\n";
		}
	}

	// Complete with label for function name

	string += "\tlabel=\"" + m_block->GetName() + "\"\n";
	string += "}";

	return string;
}
 
}
}
