#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Data/Modifiers/ConvertFlushSubnormalModifier.h"
#include "PTX/Tree/Instructions/Data/Modifiers/ConvertRoundingModifier.h"
#include "PTX/Tree/Instructions/Data/Modifiers/ConvertSaturateModifier.h"

namespace PTX {

DispatchInterface_2(ConvertInstruction)

template<class D, class S, bool Assert = true>
class ConvertInstruction : DispatchInherit(ConvertInstruction), public InstructionBase_1<D, S>, public ConvertRoundingModifier<D, S>, public ConvertFlushSubnormalModifier<D, S>, public ConvertSaturateModifier<D, S>
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

	// Formatting

	static std::string Mnemonic() { return "cvt"; }

	std::string GetOpCode() const override
	{
		std::string code = Mnemonic();
		if constexpr(ConvertRoundingModifier<D, S>::Enabled)
		{
			code += ConvertRoundingModifier<D, S>::GetOpCodeModifier();
		}
		if constexpr(ConvertFlushSubnormalModifier<D, S>::Enabled)
		{
			code += ConvertFlushSubnormalModifier<D, S>::GetOpCodeModifier();
		}
		if constexpr(ConvertSaturateModifier<D, S>::Enabled)
		{
			code += ConvertSaturateModifier<D, S>::GetOpCodeModifier();
		}
		return code + D::Name() + S::Name();
	}

	// Visitors
	
	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type1(D);
	DispatchMember_Type2(S);
};

}
