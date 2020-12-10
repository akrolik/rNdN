#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface(AbsoluteInstruction)

template<class T, bool Assert = true>
class AbsoluteInstruction : DispatchInherit(AbsoluteInstruction), public InstructionBase_1<T>, public FlushSubnormalModifier<T>
{
public:
	REQUIRE_TYPE_PARAM(AbsoluteInstruction,
		REQUIRE_EXACT(T,
			Int16Type, Int32Type, Int64Type,
			Float32Type, Float64Type
		)
	);

	using InstructionBase_1<T>::InstructionBase_1;

	// Formatting

	static std::string Mnemonic() { return "abs"; }
  	
	std::string GetOpCode() const override
	{
		if constexpr(FlushSubnormalModifier<T>::Enabled)
		{
			return Mnemonic() + FlushSubnormalModifier<T>::GetOpCodeModifier() + T::Name();
		}
		else
		{
			return Mnemonic() + T::Name();
		}
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

DispatchImplementation(AbsoluteInstruction)

}
