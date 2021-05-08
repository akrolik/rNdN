#pragma once

#include "SASS/Tree/Instructions/PredicatedInstruction.h"

#include "SASS/Tree/BinaryUtils.h"

namespace SASS {

class MEMBARInstruction : public PredicatedInstruction
{
public:
	enum class Mode : std::uint64_t {
		CTA = 0x0000000000000000,
		GL  = 0x0000000000000100,
		SYS = 0x0000000000000200
	};

	MEMBARInstruction(Mode mode) : m_mode(mode) {}

	// Properties

	Mode GetMode() const { return m_mode; }
	void SetMode(Mode mode) { m_mode = mode; }

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
		switch (m_mode)
		{
			case Mode::CTA: return ".CTA";
			case Mode::GL: return ".GL";
			case Mode::SYS: return ".SYS";
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
		return BinaryUtils::OpModifierFlags(m_mode);
	}

	// Hardware properties

	HardwareClass GetHardwareClass() const override { return HardwareClass::x32; }

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }

private:
	Mode m_mode;
};

}
