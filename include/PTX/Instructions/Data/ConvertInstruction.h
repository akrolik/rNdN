#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Data/Modifiers/ConvertFlushSubnormalModifier.h"
#include "PTX/Instructions/Data/Modifiers/ConvertRoundingModifier.h"
#include "PTX/Instructions/Data/Modifiers/ConvertSaturateModifier.h"

#include "PTX/Type.h"

namespace PTX {

template<class D, class S, bool Assert = true>
class ConvertInstruction : public InstructionBase_1<D, S>, public ConvertRoundingModifier<D, S>, public ConvertFlushSubnormalModifier<D, S>, public ConvertSaturateModifier<D, S>
{
public:
	REQUIRE_TYPE_PARAMS(ConvertInstruction,
		REQUIRE_EXACT(D,
			Int8Type, Int16Type, Int32Type, Int64Type,
			UInt8Type, UInt16Type, UInt32Type, UInt64Type,
			Float16Type, Float32Type, Float64Type
		),
		REQUIRE_EXACT(S,
			Int8Type, Int16Type, Int32Type, Int64Type,
			UInt8Type, UInt16Type, UInt32Type, UInt64Type,
			Float16Type, Float32Type, Float64Type
		)
	);

	using InstructionBase_1<D, S>::InstructionBase_1;

	static std::string Mnemonic() { return "cvt"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic();
		if constexpr(ConvertRoundingModifier<D, S>::Enabled)
		{
			code += ConvertRoundingModifier<D, S>::OpCodeModifier();
		}
		if constexpr(ConvertFlushSubnormalModifier<D, S>::Enabled)
		{
			code += ConvertFlushSubnormalModifier<D, S>::OpCodeModifier();
		}
		if constexpr(ConvertSaturateModifier<D, S>::Enabled)
		{
			code += ConvertSaturateModifier<D, S>::OpCodeModifier();
		}
		return code + D::Name() + S::Name();
	}
};

}
