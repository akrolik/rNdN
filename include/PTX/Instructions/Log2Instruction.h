#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class Log2Instruction : public PredicatedInstruction
{
	REQUIRE_TYPE(Log2Instruction, Float32Type);
public:
	Log2Instruction(Register<Float32Type> *destination, Operand<Float32Type> *source) : m_destination(destination), m_source(source) {}

	void SetFlushSubNormal(bool flush) { m_flush = flush; }

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "lg2.approx.ftz" + Float32Type::Name();
		}
		return "lg2.approx" + Float32Type::Name();
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
