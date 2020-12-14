#pragma once

#include "SASS/Operands/Operand.h"

#include "SASS/Operands/Register.h"

#include "Utils/Format.h"

namespace SASS {

class Address : public Operand
{
public:
	Address(Register *base, std::int32_t offset = 0) : m_base(base), m_offset(offset) {}

	// Properties

	const Register *GetBase() const { return m_base; }
	Register *GetBase() { return m_base; }
	void SetBase(Register *base) { m_base = base; }

	std::int32_t GetOffset() const { return m_offset; }
	void SetOffset(std::int32_t offset) { m_offset = offset; }

	// Formatting

	std::string ToString() const override
	{
		if (m_offset != 0)
		{
			return "[" + m_base->ToString() + "+" + Utils::Format::HexString(m_offset) + "]";
		}
		return "[" + m_base->ToString() + "]";
	}

	// Binary

	std::uint64_t ToBinary() const override
	{
		// Binary handled by the instruction, as it is instruction specific

		return 0x0;
	}

private:
	Register *m_base = nullptr;
	std::int32_t m_offset = 0;
};

}
