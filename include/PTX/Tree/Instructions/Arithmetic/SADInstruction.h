#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Assert = true>
class SADInstruction : public InstructionBase_2<T>
{
public:
	REQUIRE_TYPE_PARAM(SADInstruction,
		REQUIRE_EXACT(T,
			Int16Type, Int32Type, Int64Type,
			UInt16Type, UInt32Type, UInt64Type
		)
	);

	using InstructionBase_2<T>::InstructionBase_2;

	static std::string Mnemonic() { return "sad"; }

	std::string OpCode() const override
	{
		return Mnemonic() + T::Name();
	}
};

}
