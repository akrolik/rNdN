#pragma once

#include "SASS/Traversal/Visitor.h"

#include "SASS/Tree/Tree.h"

namespace SASS {
namespace Transformation {

class RedundantCodeElimination : public Visitor
{
public:
	void Transform(Function *function);

	void Visit(MOVInstruction *instruction) override;

private:
	bool m_redundant = false;
};

}
}
