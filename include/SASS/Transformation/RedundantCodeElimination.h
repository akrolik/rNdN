#pragma once

#include "SASS/Traversal/Visitor.h"

#include "SASS/Tree/Tree.h"

namespace SASS {
namespace Transformation {

class RedundantCodeElimination : public Visitor
{
public:
	void Transform(Function *function);

	void Visit(Maxwell::MOVInstruction *instruction) override;
	void Visit(Volta::MOVInstruction *instruction) override;

private:
	bool CheckRedundant(const Register *destination, const Composite *source) const;
	bool m_redundant = false;
};

}
}
