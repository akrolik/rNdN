#pragma once

#include "SASS/Operands/Immediate.h"

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
		//TODO: Float string
		// return Utils::Format::HexString(m_value);
	}

	// Binary

	std::uint64_t ToBinary() const override
	{
		//TODO: Float to binary
		// return m_value;
	}

private:
	float m_value = 0.0f;
};

}
