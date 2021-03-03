#pragma once

namespace PTX {
namespace Analysis {

class StructureNode;

class BranchStructure;
class ExitStructure;
class LoopStructure;
class SequenceStructure;

class StructuredGraphVisitor
{
public:
	virtual void Visit(StructureNode *node);

	virtual void Visit(BranchStructure *structure);
	virtual void Visit(ExitStructure *structure);
	virtual void Visit(LoopStructure *structure);
	virtual void Visit(SequenceStructure *structure);
};

}
}
