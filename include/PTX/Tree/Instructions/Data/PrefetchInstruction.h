#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/StateSpace.h"
#include "PTX/Tree/Type.h"
#include "PTX/Tree/Operands/Address/Address.h"
#include "PTX/Tree/Operands/Address/DereferencedAddress.h"

namespace PTX {

DispatchInterface_Data(PrefetchInstruction)

template<Bits B, class T, class S, bool Assert = true>
class PrefetchInstruction : DispatchInherit(PrefetchInstruction), public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(PrefetchInstruction,
		REQUIRE_BASE(T, ValueType)
	);
	REQUIRE_SPACE_PARAM(PrefetchInstruction,
		REQUIRE_EXACT(S, GlobalSpace, LocalSpace)
	);

	enum class Level {
		L1,
		L2
	};

	static std::string LevelString(Level level)
	{
		switch (level)
		{
			case Level::L1:
				return ".L1";
			case Level::L2:
				return ".L2";
		}
		return ".<unknown>";
	}

	PrefetchInstruction(Address<B, T, S> *address, Level level) : m_address(address), m_level(level) {}

	// Analysis properties

	bool HasSideEffect() const override { return true; }

	// Properties

	const Address<B, T, S> *GetAddress() const { return m_address; }
	Address<B, T, S> *GetAddress() { return m_address; }
	void SetAddress(Address<B, T, S> *address) { m_address = address; }

	Level GetLevel() const { return m_level; }
	void SetLevel(Level level) { m_level = level; }

	// Formatting

	static std::string Mnemonic() { return "prefetch"; }

	std::string GetOpCode() const override
	{
		std::string code = Mnemonic();
		if constexpr(!std::is_same<S, AddressableSpace>::value)
		{
			code += S::Name();
		}
		return code + LevelString(m_level);
	}

	std::vector<const Operand *> GetOperands() const override
	{
		return { new DereferencedAddress<B, T, S>(m_address) };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { new DereferencedAddress<B, T, S>(m_address) };
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Bits(B);
	DispatchMember_Type(T);
	DispatchMember_Space(S);

	Address<B, T, S> *m_address = nullptr;
	Level m_level;
};

template<class T, class S>
using Prefetch32Instruction = PrefetchInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using Prefetch64Instruction = PrefetchInstruction<Bits::Bits64, T, S>;

}
