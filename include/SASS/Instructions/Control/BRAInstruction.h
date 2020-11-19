#pragma once

#include "SASS/Instructions/PredicatedInstruction.h"

#include "SASS/BasicBlock.h"
#include "SASS/BinaryUtils.h"

#include "Utils/Format.h"

namespace SASS {

class BRAInstruction : public PredicatedInstruction
{
public:
	enum Flags : std::uint64_t {
		None = 0,
		U    = 0x0000000000000080
	};

	SASS_FLAGS_FRIEND()

	BRAInstruction(const BasicBlock *target, Flags flags = Flags::None) : m_target(target), m_flags(flags) {}

	// Properties

	const BasicBlock *GetTarget() const { return m_target; }
	void SetTarget(const BasicBlock *target) { m_target = target; }

	std::size_t GetTargetAddress() const { return m_targetAddress; }
	std::size_t GetCurrentAddress() const { return m_currentAddress; }

	void SetTargetAddress(std::size_t targetAddress, std::size_t currentAddress)
	{
		m_targetAddress = targetAddress;
		m_currentAddress = currentAddress;
	}

	Flags GetFlags() const { return m_flags; }
	void SetFlags(Flags flags) { m_flags = flags; }

	// Formatting

	std::string OpCode() const override { return "BRA"; }

	std::string OpModifiers() const override
	{
		if (m_flags & Flags::U)
		{
			return ".U";
		}
		return "";
	}

	std::string Operands() const override
	{
		return "`(" + m_target->GetName() + ") [" + Utils::Format::HexString(m_targetAddress, 4) + "]";
	}

	// Binary

	std::uint64_t BinaryOpCode() const override
	{
		return 0xe24000000000000f;
	}

	std::uint64_t BinaryOpModifiers() const override
	{
		return BinaryUtils::OpModifierFlags(m_flags);
	}

	std::uint64_t BinaryOperands() const override
	{
		// Represent the branch target as an 3-byte relative address, after(!) the current instructionn

		auto relativeAddress = m_targetAddress - (m_currentAddress + sizeof(std::uint64_t));
		return BinaryUtils::Format(relativeAddress, 20, 0xffffff);
	}

private:
	const BasicBlock *m_target = nullptr;

	std::size_t m_targetAddress = 0;
	std::size_t m_currentAddress = 0;

	Flags m_flags = Flags::None;
};

SASS_FLAGS_INLINE(BRAInstruction)

}
