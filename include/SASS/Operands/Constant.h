#pragma once

#include "SASS/Operands/Composite.h"

#include "Utils/Format.h"

namespace SASS {

class Constant : public Composite
{
public:
	Constant(std::uint32_t bank, std::uint32_t address) : m_bank(bank), m_address(address) {}

	// Properties

	std::uint32_t GetBank() const { return m_bank; }
	void SetBank(std::uint32_t bank) { m_bank = bank; }

	std::uint32_t GetAddress() const { return m_address; }
	void SetAddress(std::uint32_t address) { m_address = address; }

	// Formatting

	std::string ToString() const override
	{
		return "c[" + Utils::Format::HexString(m_bank) + "][" + Utils::Format::HexString(m_address) + "]";
	}

	// Binary

	std::uint64_t ToBinary() const override
	{
		return (m_bank << 19) + (m_address / sizeof(std::uint32_t));
	}

private:
	std::uint32_t m_bank = 0;
	std::uint32_t m_address = 0;
};

}
