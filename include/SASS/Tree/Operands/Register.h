#pragma once

#include "SASS/Tree/Operands/Composite.h"

namespace SASS {

class Register : public Composite
{
public:
	constexpr static std::uint8_t ZeroIndex = 255;

	Register(std::uint8_t value) : m_value(value) {}
	
	// Properties

	std::uint8_t GetValue() const { return m_value; }
	void SetValue(std::uint8_t value) { m_value = value; }

	// Formatting

	std::string ToString() const override
	{
		if (m_value == ZeroIndex)
		{
			return "RZ";
		}
		return "R" + std::to_string(m_value);
	}

	// Binary

	std::uint64_t ToBinary() const override
	{
		return m_value;
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	
private:
	std::uint8_t m_value = 0;
};

static Register *RZ = new Register(Register::ZeroIndex);

}
