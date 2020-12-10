#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Tree/Instructions/Modifiers/HalfModifier.h"
#include "PTX/Tree/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Tree/Instructions/Modifiers/SaturateModifier.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface(MultiplyInstruction)

template<class T, bool Assert = true>
class MultiplyInstruction : DispatchInherit(MultiplyInstruction), public InstructionBase_2<T>, public HalfModifier<T>, public RoundingModifier<T>, public FlushSubnormalModifier<T>, public SaturateModifier<T>
{
public:
	REQUIRE_TYPE_PARAM(MultiplyInstruction,
		REQUIRE_EXACT(T,
			Int16Type, Int32Type, Int64Type,
			UInt16Type, UInt32Type, UInt64Type,
			Float16Type, Float16x2Type, Float32Type, Float64Type
		)
	);

	template <class T1 = T, class = typename std::enable_if_t<HalfModifier<T1>::Enabled>>
	MultiplyInstruction(Register<T> *destination, TypedOperand<T> *sourceA, TypedOperand<T> *sourceB, typename HalfModifier<T1>::Half half)
		: InstructionBase_2<T>(destination, sourceA, sourceB), HalfModifier<T>(half) {} 

	template <class T1 = T, class = typename std::enable_if_t<!HalfModifier<T1>::Enabled>>
	MultiplyInstruction(Register<T> *destination, TypedOperand<T> *sourceA, TypedOperand<T> *sourceB)
		: InstructionBase_2<T>(destination, sourceA, sourceB) {}

	// Formatting

	static std::string Mnemonic() { return "mul"; }

	std::string GetOpCode() const override
	{
		std::string code = Mnemonic();
		if constexpr(HalfModifier<T>::Enabled)
		{
			code += HalfModifier<T>::GetOpCodeModifier();
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
			code += SaturateModifier<T>::GetOpCodeModifier();
		}
		return code + T::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

DispatchImplementation(MultiplyInstruction)
 
}
