#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Modifiers/CarryModifier.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Tree/Instructions/Modifiers/HalfModifier.h"
#include "PTX/Tree/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Tree/Instructions/Modifiers/SaturateModifier.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface(MADInstruction)

template<class T, bool Assert = true>
class MADInstruction : DispatchInherit(MADInstruction), public InstructionBase_3<T>, public HalfModifier<T>, public RoundingModifier<T>, public FlushSubnormalModifier<T>, public SaturateModifier<T>, public CarryModifier<T>
{
public:
	REQUIRE_TYPE_PARAM(MADInstruction,
		REQUIRE_EXACT(T,
			Int16Type, Int32Type, Int64Type,
			UInt16Type, UInt32Type, UInt64Type,
			Float32Type, Float64Type
		)
	);

	template <class T1 = T, class = typename std::enable_if_t<HalfModifier<T1>::Enabled>>
	MADInstruction(const Register<T> *destination, const TypedOperand<T> *sourceA, const TypedOperand<T> *sourceB, const TypedOperand<T> *sourceC, typename HalfModifier<T1>::Half half) : InstructionBase_3<T>(destination, sourceA, sourceB, sourceC), HalfModifier<T>(half) {} 

	template <class T1 = T, class = typename std::enable_if_t<!HalfModifier<T1>::Enabled>>
	MADInstruction(const Register<T> *destination, const TypedOperand<T> *sourceA, const TypedOperand<T> *sourceB, const TypedOperand<T> *sourceC) : InstructionBase_3<T>(destination, sourceA, sourceB, sourceC) {}

	static std::string Mnemonic() { return "mad"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic();
		if constexpr(CarryModifier<T>::Enabled)
		{
			code += CarryModifier<T>::OpCodeModifier();
		}
		if constexpr(HalfModifier<T>::Enabled)
		{
			code += HalfModifier<T>::OpCodeModifier();
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
			// Only applies in .hi mode
			if constexpr(HalfModifier<T>::Enabled)
			{
				if (HalfModifier<T>::GetUpper())
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
			}
			else
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
		}
		return code + T::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

DispatchImplementation(MADInstruction)
 
}
