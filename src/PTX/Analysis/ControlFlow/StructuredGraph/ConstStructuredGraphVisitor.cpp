#include "PTX/Analysis/ControlFlow/StructuredGraph/ConstStructuredGraphVisitor.h"

#include "PTX/Analysis/ControlFlow/StructuredGraph/StructuredGraph.h"

namespace PTX {
namespace Analysis {

void ConstStructuredGraphVisitor::Visit(const StructureNode *structure)
{

}

void ConstStructuredGraphVisitor::Visit(const BranchStructure *structure)
{
	Visit(static_cast<const StructureNode *>(structure));
}

void ConstStructuredGraphVisitor::Visit(const ExitStructure *structure)
{
	Visit(static_cast<const StructureNode *>(structure));
}

void ConstStructuredGraphVisitor::Visit(const LoopStructure *structure)
{
	Visit(static_cast<const StructureNode *>(structure));
}

void ConstStructuredGraphVisitor::Visit(const SequenceStructure *structure)
{
	Visit(static_cast<const StructureNode *>(structure));
}

}
}
