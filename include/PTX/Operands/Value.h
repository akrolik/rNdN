#pragma once

#include <iomanip>
#include <sstream>

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
		if constexpr(is_float_type<T>::value)
		{
			std::stringstream stream;
			if (sizeof(typename T::SystemType) == 4)
			{
				stream << "0F";
			}
			else
			{
				stream << "0D";
			}
			stream << std::hex << std::setfill('0');

			auto bytes = reinterpret_cast<const unsigned char *>(&m_value);
			for (int i = sizeof(typename T::SystemType) - 1; i >= 0; --i)
			{
				stream << std::setw(2) << (unsigned int)bytes[i];
			}
			return stream.str();
		}
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

using BoolValue = Value<PredicateType>;

template<Bits B>
using IntValue = Value<IntType<B>>;

using Int8Value = Value<Int8Type>;
using Int16Value = Value<Int16Type>;
using Int32Value = Value<Int32Type>;
using Int64Value = Value<Int64Type>;

template<Bits B>
using UIntValue = Value<UIntType<B>>;

using UInt8Value = Value<UInt8Type>;
using UInt16Value = Value<UInt16Type>;
using UInt32Value = Value<UInt32Type>;
using UInt64Value = Value<UInt64Type>;

template<Bits B>
using FloatValue = Value<FloatType<B>>;

using Float16Value = Value<Float16Type>;
using Float32Value = Value<Float32Type>;
using Float64Value = Value<Float64Type>;

}
