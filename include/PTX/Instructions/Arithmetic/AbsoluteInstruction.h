#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class AbsoluteInstruction : public InstructionBase<T, 1>
{
	REQUIRE_TYPE(AbsoluteInstruction, ScalarType);
	DISABLE_TYPE(AbsoluteInstruction, Int8Type);
	DISABLE_TYPE(AbsoluteInstruction, Float16Type);
	DISABLE_TYPES(AbsoluteInstruction, UIntType);
public:
	using InstructionBase<T, 1>::InstructionBase;

	std::string OpCode() const
	{
		return "abs" + T::Name();
	}
};

template<>
class AbsoluteInstruction<Float32Type> : public InstructionBase<Float32Type, 1>
{
public:
	using InstructionBase<Float32Type, 1>::InstructionBase;

	void SetFlushSubNormal(bool flush) { m_flush = flush; }

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "abs.ftz" + Float32Type::Name();
		}
		return "abs" + Float32Type::Name();
	}

private:
	bool m_flush = false;
};

}
