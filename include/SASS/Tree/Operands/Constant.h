#pragma once

#include "SASS/Tree/Operands/Composite.h"

#include "Utils/Format.h"

namespace SASS {

class Constant : public Composite
{
public:
	Constant(std::uint32_t bank, std::uint32_t address) : Composite(Operand::Kind::Constant), m_bank(bank), m_address(address) {}

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
		return (m_bank << 14) + (m_address / sizeof(std::uint32_t));
	}

	// Visitors

	void Accept(Visitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstVisitor& visitor) const override { visitor.Visit(this); }
	
private:
	std::uint32_t m_bank = 0;
	std::uint32_t m_address = 0;
};

}
