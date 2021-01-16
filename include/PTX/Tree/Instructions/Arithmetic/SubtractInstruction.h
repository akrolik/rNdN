#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Modifiers/CarryModifier.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Tree/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Tree/Instructions/Modifiers/SaturateModifier.h"

namespace PTX {

DispatchInterface(SubtractInstruction)

template<class T, bool Assert = true>
class SubtractInstruction : DispatchInherit(SubtractInstruction), public InstructionBase_2<T>, public SaturateModifier<T>, public RoundingModifier<T>, public FlushSubnormalModifier<T>, public CarryModifier<T>
{
public:
	REQUIRE_TYPE_PARAM(SubtractInstruction,
		REQUIRE_EXACT(T,
			Int16Type, Int32Type, Int64Type,
			UInt16Type, UInt32Type, UInt64Type,
			Float16Type, Float16x2Type, Float32Type, Float64Type
		)
	);

	using InstructionBase_2<T>::InstructionBase_2;

	// Formatting

	static std::string Mnemonic() { return "sub"; }

	std::string GetOpCode() const override
	{
		std::string code = Mnemonic();
		if constexpr(CarryModifier<T>::Enabled)
		{
			code += CarryModifier<T>::GetOpCodeModifier();
		}
		if constexpr(RoundingModifier<T>::Enabled)
		{
			code += RoundingModifier<T>::GetOpCodeModifier();
		}
		if constexpr(FlushSubnormalModifier<T>::Enabled)
		{
			code += FlushSubnormalModifier<T>::GetOpCodeModifier();
		}
		if constexpr(SaturateModifier<T>::Enabled)
		{
			if constexpr(CarryModifier<T>::Enabled)
			{
				if (!CarryModifier<T>::IsActive())
				{
					code += SaturateModifier<T>::GetOpCodeModifier();
				}
			}
			else
			{
				code += SaturateModifier<T>::GetOpCodeModifier();
			}
		}
		return code + T::Name();
	}

	// Visitors

	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

DispatchImplementation(SubtractInstruction)
 
}
