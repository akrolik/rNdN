#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class NegateInstruction : public PredicatedInstruction
{
	REQUIRE_TYPE(NegateInstruction, ScalarType);
	DISABLE_TYPE(NegateInstruction, Int8Type);
	DISABLE_TYPES(NegateInstruction, UIntType);
	DISABLE_TYPE(NegateInstruction, Float16Type); //TODO: Missing from PTX specification
public:
	NegateInstruction(Register<T> *destination, Operand<T> *source) : m_destination(destination), m_source(source) {}

	std::string OpCode() const
	{
		return "neg" + T::Name();
	}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}

private:
	Register<T> *m_destination = nullptr;
	Operand<T> *m_source = nullptr;
};

template<>
class NegateInstruction<Float32Type> : public PredicatedInstruction
{
public:
	NegateInstruction(Register<Float32Type> *destination, Operand<Float32Type> *source) : m_destination(destination), m_source(source) {}

	void SetFlushSubNormal(bool flush) { m_flush = flush; }

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "neg.ftz" + Float32Type::Name();
		}
		return "neg" + Float32Type::Name();
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
