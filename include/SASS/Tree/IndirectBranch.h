#pragma once

#include <string>

#include "SASS/Tree/Node.h"

#include "SASS/Tree/Instructions/Instruction.h"

namespace SASS {

class IndirectBranch : public Node
{
public:
	IndirectBranch(const Instruction *branch, const std::string& target) : m_branch(branch), m_target(target) {}
	
	// Properties

	const Instruction *GetBranch() const { return m_branch; }
	void SetBranch(const Instruction *branch) { m_branch = branch; }

	const std::string& GetTarget() const { return m_target; }
	void SetTarget(const std::string& target) { m_target = target; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }
	
private:
	const Instruction *m_branch = nullptr;
	std::string m_target;
};

}
