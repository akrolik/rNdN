#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class ReciprocalRootInstruction : public PredicatedInstruction
{
	REQUIRE_TYPES(ReciprocalRootInstruction, FloatType);
	DISABLE_TYPE(ReciprocalRootInstruction, Float16Type);
public:
	ReciprocalRootInstruction(Register<T> *destination, Operand<T> *source) : m_destination(destination), m_source(source) {}

	void SetFlushSubNormal(bool flush) { m_flush = flush; }

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "rsqrt.approx.ftz" + T::Name();
		}
		return "rsqrt.approx" + T::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}
private:
	Register<T> *m_destination = nullptr;
	Operand<T> *m_source = nullptr;

	bool m_flush = false;
};

}
