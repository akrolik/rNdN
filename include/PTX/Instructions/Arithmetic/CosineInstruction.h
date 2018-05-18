#pragma once

#include "PTX/Instructions/InstructionBase.h"

namespace PTX {

class CosineInstruction : public InstructionBase<Float32Type, 1>
{
public:
	using InstructionBase<Float32Type, 1>::InstructionBase;

	void SetFlushSubNormal(bool flush) { m_flush = flush; }

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "cos.approx.ftz" + Float32Type::Name();
		}
		return "cos.approx" + Float32Type::Name();
	}

private:
	bool m_flush = false;
};

}
