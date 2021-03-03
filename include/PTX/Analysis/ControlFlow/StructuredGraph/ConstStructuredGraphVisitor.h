#pragma once

namespace PTX {
namespace Analysis {

class StructureNode;

class BranchStructure;
class ExitStructure;
class LoopStructure;
class SequenceStructure;

class ConstStructuredGraphVisitor
{
public:
	virtual void Visit(const StructureNode *node);

	virtual void Visit(const BranchStructure *structure);
	virtual void Visit(const ExitStructure *structure);
	virtual void Visit(const LoopStructure *structure);
	virtual void Visit(const SequenceStructure *structure);
};

}
}
