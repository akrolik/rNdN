#pragma once

#include "SASS/Tree/Operands/Composite.h"

namespace SASS {

class Register : public Composite
{
public:
	constexpr static std::uint8_t ZeroIndex = 255;

	Register(std::uint8_t value, std::uint8_t range = 1) : Composite(Operand::Kind::Register), m_value(value), m_range(range) {}

	// Properties

	std::uint8_t GetValue() const { return m_value; }
	void SetValue(std::uint8_t value) { m_value = value; }

	std::uint8_t GetRange() const { return m_range; }
	void SetRange(std::uint8_t range) { m_range = range; }

	// Formatting

	std::string ToString() const override
	{
		if (m_value == ZeroIndex)
		{
			return "RZ";
		}
		return "R" + std::to_string(m_value);
	}

	std::string ToSizedString() const
	{
		if (m_range == 2)
		{
			return ToString() + ".64";
		}
		return ToString();
	}

	// Binary

	std::uint64_t ToBinary() const override
	{
		return m_value;
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }
	
private:
	std::uint8_t m_value = 0;
	std::uint8_t m_range = 0;
};

static Register *RZ = new Register(Register::ZeroIndex);

}
