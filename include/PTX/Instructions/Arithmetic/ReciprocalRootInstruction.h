#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

template<class T>
class ReciprocalRootInstruction : public InstructionBase<T, 1>
{
	REQUIRE_TYPES(ReciprocalRootInstruction, FloatType);
	DISABLE_TYPE(ReciprocalRootInstruction, Float16Type);
public:
	using InstructionBase<T, 1>::InstructionBase;

	void SetFlushSubNormal(bool flush) { m_flush = flush; }

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "rsqrt.approx.ftz" + T::Name();
		}
		return "rsqrt.approx" + T::Name();
	}

private:
	bool m_flush = false;
};

}
