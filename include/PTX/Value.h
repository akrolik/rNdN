#pragma once

#include "PTX/Operand.h"

namespace PTX {

template<typename T>
class Value : Operand<T>
{
public:
	Value(T value) : m_value(value) {}

	T GetValue() { return m_value; }
	void SetValue(T value) { m_value = value; }

private:
	T m_value;
};

}
