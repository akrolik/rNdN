#pragma once

#include "PTX/Operands/Operand.h"

namespace PTX {

class Int32Value : public Operand<Int32Type>
{
public:
	Int32Value(int32_t value) : m_value(value) {}

	int32_t GetValue() { return m_value; }
	void SetValue(int32_t value) { m_value = value; }

	std::string ToString()
	{
		return std::to_string(m_value);
	}

private:
	int32_t m_value;
};

}
