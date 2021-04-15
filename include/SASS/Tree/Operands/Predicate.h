#pragma once

#include "SASS/Tree/Operands/Operand.h"

namespace SASS {

class Predicate : public Operand
{
public:
	constexpr static std::uint8_t TrueIndex = 7;

	Predicate(std::uint8_t value) : m_value(value) {}
	
	// Properties

	std::uint8_t GetValue() const { return m_value; }
	void SetValue(std::uint8_t value) { m_value = value; }

	// Formatting

	std::string ToString() const override
	{
		if (m_value == TrueIndex)
		{
			return "PT";
		}
		return "P" + std::to_string(m_value);
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

static Predicate *PT = new Predicate(Predicate::TrueIndex);

}
