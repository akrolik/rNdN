#pragma once

#include "PTX/Operands/Operand.h"

namespace PTX {

template<class T>
class Value : public TypedOperand<T>
{
public:
	Value(typename T::SystemType value) : m_value(value) {}

	typename T::SystemType GetValue() const { return m_value; }
	void SetValue(typename T::SystemType value) { m_value = value; }

	std::string ToString() const override
	{
		return std::to_string(m_value);
	}

	json ToJSON() const override
	{
		json j;

		j["type"] = T::Name();
		j["value"] = m_value;

		return j;
	}

private:
	typename T::SystemType m_value;
};

using Int32Value = Value<Int32Type>;
using Int64Value = Value<Int64Type>;

using UInt32Value = Value<UInt32Type>;
using UInt64Value = Value<UInt64Type>;

using Float32Value = Value<Float32Type>;
using Float64Value = Value<Float64Type>;

}
