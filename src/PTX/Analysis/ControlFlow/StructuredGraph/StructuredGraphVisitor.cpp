#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraphVisitor.h"

#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraph.h"

namespace PTX {
namespace Analysis {

void StructuredGraphVisitor::Visit(StructureNode *structure)
{

}

void StructuredGraphVisitor::Visit(BranchStructure *structure)
{
	Visit(static_cast<StructureNode *>(structure));
}

void StructuredGraphVisitor::Visit(ExitStructure *structure)
{
	Visit(static_cast<StructureNode *>(structure));
}

void StructuredGraphVisitor::Visit(LoopStructure *structure)
{
	Visit(static_cast<StructureNode *>(structure));
}

void StructuredGraphVisitor::Visit(SequenceStructure *structure)
{
	Visit(static_cast<StructureNode *>(structure));
}

}
}
