#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class TestPropertyInstruction : public InstructionBase<T, 1, PredicateType>
{
	REQUIRE_TYPES(TestPropertyInstruction, FloatType);
	DISABLE_TYPE(TestPropertyInstruction, Float16Type);
	DISABLE_TYPE(TestPropertyInstruction, Float16x2Type);
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
		return ".<unknow>";
	}

	TestPropertyInstruction(Register<PredicateType> *destination, Operand<T> *source, Property property) : InstructionBase<T, 1, PredicateType>(destination, source), m_property(property) {}

	std::string OpCode() const
	{
		return "testp" + GetPropertyString(m_property) + T::Name();
	}

private:
	Property m_property;
};

}