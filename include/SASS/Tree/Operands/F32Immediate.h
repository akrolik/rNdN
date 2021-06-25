#pragma once

#include "SASS/Tree/Operands/Immediate.h"

#include "Utils/Format.h"

namespace SASS {

class F32Immediate : public Immediate
{
public:
	F32Immediate(float value) : m_value(value) {}

	// Properties

	float GetValue() const { return m_value; }
	void SetValue(float value) { m_value = value; }

	// Formatting

	std::string ToString() const override
	{
		auto inf = std::numeric_limits<float>::infinity();
		if (m_value == inf)
		{
			return "INF";
		}
		return std::to_string(m_value);
	}

	// Binary

	std::uint64_t ToBinary() const override
	{
		return reinterpret_cast<const std::uint64_t&>(m_value);
	}

	std::uint64_t ToBinary(std::uint8_t truncate) const override
	{
		// Composite truncated binary representation:
		//  - Sign removed
		//  - Truncated by shifting right
		// The sign bit is always zero, meaning we have truncate - 1 bits of data

		auto value = (m_value < 0) ? -m_value : m_value;
		return reinterpret_cast<const std::uint64_t&>(value) >> (32 - truncate);
	}

	bool GetOpModifierNegate() const override
	{
		return (m_value < 0);
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }
	
private:
	float m_value = 0.0f;
};

}
