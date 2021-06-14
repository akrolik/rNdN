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

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }
	
private:
	float m_value = 0.0f;
};

}
