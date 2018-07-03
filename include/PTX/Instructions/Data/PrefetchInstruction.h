#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/StateSpace.h"
#include "PTX/Type.h"
#include "PTX/Operands/Address/Address.h"
#include "PTX/Operands/Address/DereferencedAddress.h"

namespace PTX {

template<Bits B, class T, class S>
class PrefetchInstruction : public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(PrefetchInstruction,
		REQUIRE_BASE(T, DataType)
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

	PrefetchInstruction(const Address<B, T, S> *address, Level level) : m_address(address), m_level(level) {}

	std::string OpCode() const override
	{
		std::string code = "prefetch";
		if constexpr (!std::is_same<S, AddressableSpace>::value)
		{
			code += S::Name();
		}
		return code + LevelString(m_level);
	}

	std::vector<const Operand *> Operands() const override
	{
		return { new DereferencedAddress<B, T, S>(m_address) };
	}

private:
	const Address<B, T, S> *m_address = nullptr;
	Level m_level;
};

template<class T, class S>
using Prefetch32Instruction = PrefetchInstruction<Bits::Bits32, T, S>;
template<class T, class S>
using Prefetch64Instruction = PrefetchInstruction<Bits::Bits64, T, S>;

}
