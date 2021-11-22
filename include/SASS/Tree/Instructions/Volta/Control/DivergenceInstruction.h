#pragma once

#include "SASS/Tree/Instructions/Volta/Control/ControlInstruction.h"
#include "SASS/Tree/Instructions/Volta/BinaryUtils.h"

namespace SASS {
namespace Volta {

class DivergenceInstruction : public ControlInstruction
{
public:
	DivergenceInstruction(const std::string& target, Predicate *controlPredicate = nullptr, bool negatePredicate = false)
		: ControlInstruction(controlPredicate, negatePredicate), m_target(target) {}

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
		std::string code;

		// Optional control predicate spacing

		code += ControlInstruction::Operands();
		if (code.length() > 0)
		{
			code += ", ";
		}

		// Absolute target address

		code += "`(" + m_target + ") [" + Utils::Format::HexString(m_targetAddress, 4) + "]";

		return code;
	}

	// Binary

	std::uint64_t ToBinary() const override
	{
		auto code = ControlInstruction::ToBinary();

		// Represent the branch target as a realtive address in two parts, after(!) the current instructionn

		auto relativeAddress = (m_targetAddress - (m_currentAddress + 2 * sizeof(std::uint64_t))) / 4;
		code |= BinaryUtils::Format(relativeAddress, 34, 0x3fffffff);

		return code;
	}

	std::uint64_t ToBinaryHi() const override
	{
		auto code = ControlInstruction::ToBinaryHi();

		// Represent the branch target as a realtive address in two parts, after(!) the current instructionn

		auto relativeAddress = (m_targetAddress - (m_currentAddress + 2 * sizeof(std::uint64_t))) / 4;
		code |= BinaryUtils::Format(relativeAddress >> 30, 0, 0x3ffff);

		return code;
	}

private:
	std::string m_target;

	std::size_t m_targetAddress = 0;
	std::size_t m_currentAddress = 0;
};

}
}
