#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class CosineInstruction : public PredicatedInstruction
{
	REQUIRE_TYPE(CosineInstruction, Float32Type);
public:
	CosineInstruction(Register<Float32Type> *destination, Operand<Float32Type> *source) : m_destination(destination), m_source(source) {}

	void SetFlushSubNormal(bool flush) { m_flush = flush; }

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "cos.approx.ftz" + Float32Type::Name();
		}
		return "cos.approx" + Float32Type::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}
private:
	Register<Float32Type> *m_destination = nullptr;
	Operand<Float32Type> *m_source = nullptr;

	bool m_flush = false;
};

}
