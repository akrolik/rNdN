#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface_2(SignSelectInstruction)

template<class D, class T, bool Assert = true>
class SignSelectInstruction : DispatchInherit(SignSelectInstruction), public InstructionBase_3<D, D, D, T>, public FlushSubnormalModifier<T>
{
public:
	REQUIRE_TYPE_PARAMS(SetInstruction,
		REQUIRE_EXACT(D,
			Bit16Type, Bit32Type, Bit64Type,
			Int16Type, Int32Type, Int64Type,
			UInt16Type, UInt32Type, UInt64Type,
			Float32Type, Float64Type
		),
		REQUIRE_EXACT(T,
			Int32Type, Float32Type
		)
	);

	using InstructionBase_3<D, D, D, T>::InstructionBase_3;

	// Formatting

	static std::string Mnemonic() { return "slct"; }

	std::string GetOpCode() const override
	{
		std::string code = Mnemonic();
		if constexpr(FlushSubnormalModifier<T>::Enabled)
		{
			code += FlushSubnormalModifier<T>::GetOpCodeModifier();
		}
		return code + D::Name() + T::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type1(D);
	DispatchMember_Type2(T);
};

DispatchImplementation_2(SignSelectInstruction)

}
