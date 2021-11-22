#pragma once

#include "SASS/Tree/Instructions/Maxwell/Control/DivergenceInstruction.h"
#include "SASS/Tree/Instructions/Maxwell/BinaryUtils.h"

namespace SASS {
namespace Maxwell {

class BRAInstruction : public DivergenceInstruction
{
public:
	enum Flags : std::uint64_t {
		None = 0,
		U    = 0x0000000000000080
	};

	SASS_FLAGS_FRIEND()

	BRAInstruction(const std::string& target, Flags flags = Flags::None) : DivergenceInstruction(target), m_flags(flags) {}

	// Properties

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

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

	std::string OpCode() const override { return "BRA"; }

	std::string OpModifiers() const override
	{
		if (m_flags & Flags::U)
		{
			return ".U";
		}
		return "";
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xe24000000000000f;
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_flags);
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(BRAInstruction)

}
}
