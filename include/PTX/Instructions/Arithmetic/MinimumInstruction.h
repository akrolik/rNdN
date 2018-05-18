#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class MinimumInstruction : public InstructionBase<T, 2>
{
	REQUIRE_TYPE(MinimumInstruction, ScalarType);
	DISABLE_TYPE(MinimumInstruction, Int8Type);
	DISABLE_TYPE(MinimumInstruction, UInt8Type);
	DISABLE_TYPE(MinimumInstruction, Float16Type);
public:
	using InstructionBase<T, 2>::InstructionBase;

	std::string OpCode() const
	{
		return "min" + T::Name();
	}
};

template<>
class MinimumInstruction<Float32Type> : public InstructionBase<Float32Type, 2>
{
public:
	using InstructionBase<Float32Type, 2>::InstructionBase;

	void SetFlushSubNormal(bool flush) { m_flush = flush; }

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "min.ftz" + Float32Type::Name();
		}
		return "min" + Float32Type::Name();
	}

private:
	bool m_flush = false;
};

}
