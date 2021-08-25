#include "SASS/Analysis/Dependency/BlockDependencyGraph.h"

#include "Libraries/robin_hood.h"

#include "SASS/Utils/PrettyPrinter.h"

namespace SASS {
namespace Analysis {

std::string BlockDependencyGraph::ToDOTString() const
{
	robin_hood::unordered_map<Instruction *, unsigned int> indexMap;
	auto index = 1u;

 	// Construct the DOT surrounding structure

	auto string = "digraph dependency_" + m_block->GetName() + " {\n";

	// Add all CFG nodes (basic blocks) in the graph

	for (const auto& node : m_nodes)
	{
		auto instruction = node.GetInstruction();
		indexMap[instruction] = index;

		auto name = "n_" + std::to_string(index);
		auto label = SASS::PrettyPrinter::PrettyString(instruction);

		string += "\t" + name + "[label=\"" + label + "\", shape=plaintext]\n"; 

		index++;
	}
	string += "\n";

	// Add all edges in the graph

	for (const auto& node : m_nodes)
	{
		auto instruction = node.GetInstruction();
		for (const auto& edgeRef : node.GetOutgoingEdges())
		{
			const auto& edge = edgeRef.get();

			auto successor = edge.GetEndInstruction();

			auto nodeIndex = std::to_string(indexMap[instruction]);
			auto successorIndex = std::to_string(indexMap[successor]);

			string += "\tn_" + nodeIndex + " -> n_" + successorIndex + " [style=bold, label=\"";

			auto dependencies = edge.GetDependencies();
			if (dependencies & DependencyKind::ReadWrite)
			{
				string += "a";
			}
			if (dependencies & DependencyKind::WriteRead)
			{
				string += "t";
			}
			if (dependencies & DependencyKind::WriteReadPredicate)
			{
				string += "p";
			}
			if (dependencies & DependencyKind::WriteWrite)
			{
				string += "w";
			}
			string += "\"];\n";
		}
	}

	// Complete with label for function name

	string += "\tlabel=\"" + m_block->GetName() + "\"\n";
	string += "}";

	return string;
}
 
}
}
