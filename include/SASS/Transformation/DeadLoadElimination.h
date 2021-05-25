#pragma once

#include "SASS/Traversal/Visitor.h"

#include "SASS/Tree/Tree.h"

namespace SASS {
namespace Transformation {

class DeadLoadElimination : public Visitor
{
public:
	void Transform(Function *function);

	void Visit(LDGInstruction *instruction) override;
	void Visit(LDSInstruction *instruction) override;

private:
	bool m_dead = false;
};

}
}
