#pragma once

#include "SASS/Tree/Operands/Immediate.h"

#include "Utils/Format.h"

namespace SASS {

class I8Immediate : public Immediate
{
public:
	I8Immediate(std::uint8_t value) : m_value(value) {}

	// Properties

	std::uint8_t GetValue() const { return m_value; }
	void SetValue(std::uint8_t value) { m_value = value; }

	// Formatting

	std::string ToString() const override
	{
		return Utils::Format::HexString(m_value);
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
};

}
