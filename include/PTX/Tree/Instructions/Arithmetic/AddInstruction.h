#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Modifiers/CarryModifier.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Tree/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Tree/Instructions/Modifiers/SaturateModifier.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface(AddInstruction)

template<class T, bool Assert = true>
class AddInstruction : DispatchInherit(AddInstruction), public InstructionBase_2<T>, public SaturateModifier<T>, public RoundingModifier<T>, public FlushSubnormalModifier<T>, public CarryModifier<T>
{
public:
	REQUIRE_TYPE_PARAM(AddInstruction,
		REQUIRE_EXACT(T,
			Int16Type, Int32Type, Int64Type,
			UInt16Type, UInt32Type, UInt64Type,
			Float16Type, Float16x2Type, Float32Type, Float64Type
		)
	);

	using InstructionBase_2<T>::InstructionBase_2;

	static std::string Mnemonic() { return "add"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic();
		if constexpr(CarryModifier<T>::Enabled)
		{
			code += CarryModifier<T>::OpCodeModifier();
		}
		if constexpr(RoundingModifier<T>::Enabled)
		{
			code += RoundingModifier<T>::OpCodeModifier();
		}
		if constexpr(FlushSubnormalModifier<T>::Enabled)
		{
			code += FlushSubnormalModifier<T>::OpCodeModifier();
		}
		if constexpr(SaturateModifier<T>::Enabled)
		{
			if constexpr(CarryModifier<T>::Enabled)
			{
				if (!CarryModifier<T>::IsActive())
				{
					code += SaturateModifier<T>::OpCodeModifier();
				}
			}
			else
			{
				code += SaturateModifier<T>::OpCodeModifier();
			}
		}
		return code + T::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

DispatchImplementation(AddInstruction)

}
