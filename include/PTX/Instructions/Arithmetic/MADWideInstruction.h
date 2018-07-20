#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Assert = true>
class MADWideInstruction : public InstructionBase_3<typename T::WideType, T>
{
public:
	REQUIRE_TYPE_PARAM(MADWideInstruction,
		REQUIRE_EXACT(T, Int16Type, Int32Type, UInt16Type, UInt32Type)
	);

	using InstructionBase_2<typename T::WideType, T>::InstructionBase_2;

	static std::string Mnemonic() { return "mad"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ".wide" + T::Name();
	}
};

}
