#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/HalfModifier.h"
#include "PTX/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Instructions/Modifiers/SaturateModifier.h"

namespace PTX {

template<class T, bool Typecheck = true>
class MultiplyInstruction : public InstructionBase_2<T>, public HalfModifier<T>, public RoundingModifier<T>, public FlushSubnormalModifier<T>, public SaturateModifier<T>
{
public:
	REQUIRE_TYPE(MultiplyInstruction,
		Int16Type, Int32Type, Int64Type,
		UInt16Type, UInt32Type, UInt64Type,
		Float16Type, Float16x2Type, Float32Type, Float64Type
	);

	using InstructionBase_2<T>::InstructionBase_2;

	std::string OpCode() const override
	{
		std::string code = "mul";
		if constexpr(T::HalfModifier)
		{
			if (this->m_upper)
			{
				code += ".hi";
			}
			else if (this->m_lower)
			{
				code += ".lo";
			}
		}
		if constexpr(is_rounding_type<T>::value)
		{
			code += T::RoundingModeString(this->m_roundingMode);
		}
		if constexpr(T::FlushModifier)
		{
			if (this->m_flush)
			{
				code += ".ftz";
			}
		}
		if constexpr(T::SaturateModifier)
		{
			if (this->m_saturate)
			{
				code += ".sat";
			}
		}
		return code + T::Name();
	}
};

template<>
class MultiplyInstruction<Int32Type> : public InstructionBase_2<Int32Type>, public HalfModifier<Int32Type>
{
public:
	using InstructionBase_2<Int32Type>::InstructionBase_2;

	std::string OpCode() const override
	{
		std::string code = "mul";
		if (this->m_upper)
		{
			code += ".hi";
		}
		else if (this->m_lower)
		{
			code += ".lo";
		}
		return code + Int32Type::Name();
	}
};

}
