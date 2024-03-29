#pragma once

#include <iomanip>
#include <sstream>

#include "PTX/Tree/Operands/Operand.h"

#include "Utils/Format.h"

namespace PTX {

DispatchInterface(Value)

template<class T, bool Assert = true>
class Value : DispatchInherit(Value), public TypedOperand<T, Assert>
{
public:
	REQUIRE_TYPE_PARAM(Value,
		REQUIRE_BASE(T, ScalarType)
	);

	Value(typename T::SystemType value) : m_value(value) {}

	// Properties

	typename T::SystemType GetValue() const { return m_value; }
	void SetValue(typename T::SystemType value) { m_value = value; }

	// Formatting

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
		else if constexpr(is_int_type<T>::value)
		{
			return Utils::Format::HexString(m_value);
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

	// Visitors

	void Accept(OperandVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstOperandVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);

	typename T::SystemType m_value;
};

using BoolValue = Value<PredicateType>;

template<Bits B>
using BitValue = Value<BitType<B>>;

using Bit8Value = Value<Bit8Type>;
using Bit16Value = Value<Bit16Type>;
using Bit32Value = Value<Bit32Type>;
using Bit64Value = Value<Bit64Type>;

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
