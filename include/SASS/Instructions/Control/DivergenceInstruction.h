#pragma once

#include "SASS/Instructions/Instruction.h"

#include "SASS/BinaryUtils.h"

#include "Utils/Format.h"

#include <string>

namespace SASS {

class DivergenceInstruction : public Instruction
{
public:
	DivergenceInstruction(const std::string& target) : m_target(target) {}

	// Properties

	const std::string& GetTarget() const { return m_target; }
	void SetTarget(const std::string &target) { m_target = target; }

	std::size_t GetTargetAddress() const { return m_targetAddress; }
	std::size_t GetCurrentAddress() const { return m_currentAddress; }

	void SetTargetAddress(std::size_t targetAddress, std::size_t currentAddress)
	{
		m_targetAddress = targetAddress;
		m_currentAddress = currentAddress;
	}

	// Formatting

	std::string Operands() const override
	{
		return "`(" + m_target + ") [" + Utils::Format::HexString(m_targetAddress, 4) + "]";
	}

	// Binary

	std::uint64_t BinaryOperands() const override
	{
		// Represent the branch target as an 3-byte relative address, after(!) the current instructionn

		auto relativeAddress = m_targetAddress - (m_currentAddress + sizeof(std::uint64_t));
		return BinaryUtils::Format(relativeAddress, 20, 0xffffff);
	}

private:
	std::string m_target;

	std::size_t m_targetAddress = 0;
	std::size_t m_currentAddress = 0;
};

}
