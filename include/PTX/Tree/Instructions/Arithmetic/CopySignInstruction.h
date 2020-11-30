#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Assert = true>
class CopySignInstruction : public InstructionBase_2<T>
{
public:
	REQUIRE_TYPE_PARAM(CopySignInstruction,
		REQUIRE_EXACT(T,
			Float32Type, Float64Type
		)
	);

	using InstructionBase_2<T>::InstructionBase;

	static std::string Mnemonic() { return "copysign"; }
	
	std::string OpCode() const override
	{
		return Mnemonic() + T::Name();
	}
};

}
