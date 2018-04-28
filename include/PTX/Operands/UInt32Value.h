#pragma once

#include "PTX/Operand.h"

namespace PTX {

class UInt32Value : public Operand<UInt32Type>
{
public:
	UInt32Value(uint32_t value) : m_value(value) {}

	uint32_t GetValue() { return m_value; }
	void SetValue(uint32_t value) { m_value = value; }

	std::string ToString()
	{
		return std::to_string(m_value);
	}

private:
	uint32_t m_value;
};

}
