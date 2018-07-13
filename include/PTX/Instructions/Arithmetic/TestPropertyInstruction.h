#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T, bool Assert = true>
class TestPropertyInstruction : public InstructionBase_1<PredicateType, T>
{
public:
	REQUIRE_TYPE_PARAM(TestPropertyInstruction,
		REQUIRE_EXACT(T,
			Float32Type, Float64Type
		)
	);

	enum class Property {
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
			case Property::Finite:
				return ".finite";
			case Property::Infinite:
				return ".infinite";
			case Property::Number:
				return ".number";
			case Property::NaN:
				return ".notanumber";
			case Property::Normal:
				return ".normal";
			case Property::Subnormal:
				return ".subnormal";
		}
		return ".<unknown>";
	}

	TestPropertyInstruction(const Register<PredicateType> *destination, const TypedOperand<T> *source, Property property) : InstructionBase_1<PredicateType, T>(destination, source), m_property(property) {}

	static std::string Mnemonic() { return "testp"; }

	std::string OpCode() const override
	{
		return Mnemonic() + GetPropertyString(m_property) + T::Name();
	}

private:
	Property m_property;
};

}
