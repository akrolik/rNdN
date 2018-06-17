#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Typecheck = true>
class TestPropertyInstruction : public InstructionBase_1<PredicateType, T>
{
public:
	REQUIRE_TYPE(TestPropertyInstruction,
		Float32Type, Float64Type
	);

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

	TestPropertyInstruction(const Register<PredicateType> *destination, const Operand<T> *source, Property property) : InstructionBase_1<PredicateType, T>(destination, source), m_property(property) {}

	std::string OpCode() const override
	{
		return "testp" + GetPropertyString(m_property) + T::Name();
	}

private:
	Property m_property;
};

}
