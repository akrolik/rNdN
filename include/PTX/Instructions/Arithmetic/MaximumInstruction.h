#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class MaximumInstruction : public InstructionBase<T, 2>
{
	REQUIRE_TYPE(MaximumInstruction, ScalarType);
	DISABLE_TYPE(MaximumInstruction, Int8Type);
	DISABLE_TYPE(MaximumInstruction, UInt8Type);
	DISABLE_TYPE(MaximumInstruction, Float16Type);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		return "max" + T::Name();
	}
};

template<>
class MaximumInstruction<Float32Type> : public InstructionBase<Float32Type, 2>
{
public:
	using InstructionBase<Float32Type, 2>::InstructionBase;

	void SetFlushSubNormal(bool flush) { m_flush = flush; }

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "max.ftz" + Float32Type::Name();
		}
		return "max" + Float32Type::Name();
	}

private:
	bool m_flush = false;
};

}
