#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface(NegateInstruction)

template<class T, bool Assert = true>
class NegateInstruction : DispatchInherit(NegateInstruction), public InstructionBase_1<T>, public FlushSubnormalModifier<T>
{
public:
	REQUIRE_TYPE_PARAM(NegateInstruction,
		REQUIRE_EXACT(T,
			Int16Type, Int32Type, Int64Type,
			Float16Type, Float16x2Type, Float32Type, Float64Type
		)
	);

	using InstructionBase_1<T>::InstructionBase_1;

	// Formatting

	static std::string Mnemonic() { return "neg"; }

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

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

DispatchImplementation(NegateInstruction)
 
}
