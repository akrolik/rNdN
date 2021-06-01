#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"

namespace PTX {

DispatchInterface(MaximumInstruction)

template<class T, bool Assert = true>
class MaximumInstruction : DispatchInherit(MaximumInstruction), public InstructionBase_2<T>, public FlushSubnormalModifier<T>
{
public:
	REQUIRE_TYPE_PARAM(MaximumInstruction,
		REQUIRE_EXACT(T,
			Int16Type, Int32Type, Int64Type,
			UInt16Type, UInt32Type, UInt64Type,
			Float32Type, Float64Type
		)
	);

	using InstructionBase_2<T>::InstructionBase_2;

	// Analysis properties

	bool HasSideEffect() const override { return false; }

	// Formatting

	static std::string Mnemonic() { return "max"; }

	std::string GetOpCode() const override
	{
		std::string code = Mnemonic();
		if constexpr(FlushSubnormalModifier<T>::Enabled)
		{
			code += FlushSubnormalModifier<T>::GetOpCodeModifier();
		}
		return code + T::Name();
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

}
