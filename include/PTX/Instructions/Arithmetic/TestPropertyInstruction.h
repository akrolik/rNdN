#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class TestPropertyInstruction : public InstructionBase_1<PredicateType, T>
{
	REQUIRE_EXACT_TYPE_TEMPLATE(TestPropertyInstruction, FloatType);
	DISABLE_EXACT_TYPE(TestPropertyInstruction, Float16Type);
	DISABLE_EXACT_TYPE(TestPropertyInstruction, Float16x2Type);
public:
	enum Property {
		Finite,
		Infinite,
		Number,
		NaN,
		Normal,
		Subnormal
	};

	static std::string GetPropertyString(Property property)
	{
		switch (property)
		{
			case Finite:
				return ".finite";
			case Infinite:
				return ".infinite";
			case Number:
				return ".number";
			case NaN:
				return ".notanumber";
			case Normal:
				return ".normal";
			case Subnormal:
				return ".subnormal";
		}
		return ".<unknown>";
	}

	TestPropertyInstruction(Register<PredicateType> *destination, Operand<T> *source, Property property) : InstructionBase_1<PredicateType, T>(destination, source), m_property(property) {}

	std::string OpCode() const
	{
		return "testp" + GetPropertyString(m_property) + T::Name();
	}

private:
	Property m_property;
};

}
