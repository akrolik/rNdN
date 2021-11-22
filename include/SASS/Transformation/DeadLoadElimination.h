#pragma once

#include "SASS/Traversal/Visitor.h"

#include "SASS/Tree/Tree.h"

namespace SASS {
namespace Transformation {

class DeadLoadElimination : public Visitor
{
public:
	void Transform(Function *function);

	void Visit(Maxwell::LDGInstruction *instruction) override;
	void Visit(Maxwell::LDSInstruction *instruction) override;

	void Visit(Volta::LDGInstruction *instruction) override;
	void Visit(Volta::LDSInstruction *instruction) override;

private:
	bool CheckDeadLoad(const Register *destination);
	bool m_dead = false;
};

}
}
