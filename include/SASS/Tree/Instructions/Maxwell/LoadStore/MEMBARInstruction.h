#pragma once

#include "SASS/Tree/Instructions/Maxwell/PredicatedInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/BinaryUtils.h"

namespace SASS {
namespace Maxwell {

class MEMBARInstruction : public PredicatedInstruction
{
public:
	enum class Scope : std::uint64_t {
		CTA = 0x0000000000000000,
		GL  = 0x0000000000000100,
		SYS = 0x0000000000000200
	};

	MEMBARInstruction(Scope scope) : m_scope(scope) {}

	// Properties

	Scope GetScope() const { return m_scope; }
	void SetScope(Scope scope) { m_scope = scope; }

	// Operands
	
	std::vector<Operand *> GetDestinationOperands() const override
	{
		return { };
	}

	std::vector<Operand *> GetSourceOperands() const override
	{
		return { };
	}

	// Formatting

	std::string OpCode() const override { return "MEMBAR"; }

	std::string OpModifiers() const override
	{
		switch (m_scope)
		{
			case Scope::CTA: return ".CTA";
			case Scope::GL: return ".GL";
			case Scope::SYS: return ".SYS";
		}
		return "";
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xef98000000000000;
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_scope);
	}

	// Hardware properties

	InstructionClass GetInstructionClass() const override { return InstructionClass::Control; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Scope m_scope;
};

}
}
