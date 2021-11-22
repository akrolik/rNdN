#pragma once

#include "SASS/Tree/Instructions/Volta/Control/DivergenceInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

namespace SASS {
namespace Volta {

class BRAInstruction : public DivergenceInstruction
{
public:
	enum class Divergence : std::uint64_t {
		None = 0x0,
		U    = 0x1,
		DIV  = 0x2,
		CONV = 0x3
	};

	BRAInstruction(const std::string& target, Divergence divergence = Divergence::None, Predicate *controlPredicate = nullptr, bool negatePredicate = false)
		: DivergenceInstruction(target, controlPredicate, negatePredicate), m_divergence(divergence) {}

	// Properties

	Divergence GetDivergence() const { return m_divergence; }
	void SetDivergence(Divergence divergence) { m_divergence = divergence; }

	// Formatting

	std::string OpCode() const override { return "BRA"; }

	std::string OpModifiers() const override
	{
		switch (m_divergence)
		{
			case Divergence::U: return ".U";
			case Divergence::CONV: return ".CONV";
			case Divergence::DIV: return ".DIV";
		}
		return "";
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0x947;
	}

	std::uint64_t ToBinary() const override
	{
		auto code = DivergenceInstruction::ToBinary();

		// Divergence mode
		code |= BinaryUtils::Format(m_divergence, 32, 0x3);

		return code;
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Divergence m_divergence = Divergence::None;
};

}
}
