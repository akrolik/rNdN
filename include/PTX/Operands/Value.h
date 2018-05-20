#pragma once

#include "PTX/Operands/Operand.h"

namespace PTX {

template<class T>
class Value : public Operand<T>
{

};

template<>
class Value<Int32Type> : public Operand<Int32Type>
{
public:
	Value(int32_t value) : m_value(value) {}

	int32_t GetValue() const { return m_value; }
	void SetValue(int32_t value) { m_value = value; }

	std::string ToString() const
	{
		return std::to_string(m_value);
	}

private:
	int32_t m_value;
};

template<>
class Value<UInt32Type> : public Operand<UInt32Type>
{
public:
	Value(uint32_t value) : m_value(value) {}

	uint32_t GetValue() const { return m_value; }
	void SetValue(uint32_t value) { m_value = value; }

	std::string ToString() const
	{
		return std::to_string(m_value);
	}

private:
	uint32_t m_value;
};

template<>
class Value<Int64Type> : public Operand<Int64Type>
{
public:
	Value(int64_t value) : m_value(value) {}

	int64_t GetValue() const { return m_value; }
	void SetValue(int64_t value) { m_value = value; }

	std::string ToString() const
	{
		return std::to_string(m_value);
	}

private:
	int64_t m_value;
};

template<>
class Value<UInt64Type> : public Operand<UInt64Type>
{
public:
	Value(uint64_t value) : m_value(value) {}

	uint64_t GetValue() const { return m_value; }
	void SetValue(uint64_t value) { m_value = value; }

	std::string ToString() const
	{
		return std::to_string(m_value);
	}

private:
	uint64_t m_value;
};

using Int32Value = Value<Int32Type>;
using UInt32Value = Value<UInt32Type>;
using Int64Value = Value<Int64Type>;
using UInt64Value = Value<UInt64Type>;

template<>
class Value<Float32Type> : public Operand<Float32Type>
{
public:
	Value(float value) : m_value(value) {}

	float GetValue() const { return m_value; }
	void SetValue(uint64_t value) { m_value = value; }

	std::string ToString() const
	{
		return std::to_string(m_value);
	}

private:
	float m_value;
};

template<>
class Value<Float64Type> : public Operand<Float64Type>
{
public:
	Value(double value) : m_value(value) {}

	double GetValue() const { return m_value; }
	void SetValue(double value) { m_value = value; }

	std::string ToString() const
	{
		return std::to_string(m_value);
	}

private:
	double m_value;
};

using Float32Value = Value<Float32Type>;
using Float64Value = Value<Float64Type>;

}
