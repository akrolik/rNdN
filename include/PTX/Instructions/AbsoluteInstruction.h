#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T>
class AbsoluteInstruction : public PredicatedInstruction
{
	REQUIRE_TYPE(AbsoluteInstruction, ScalarType);
	DISABLE_TYPE(AbsoluteInstruction, Int8Type);
	DISABLE_TYPE(AbsoluteInstruction, Float16Type);
	DISABLE_TYPES(AbsoluteInstruction, UIntType);
public:
	AbsoluteInstruction(Register<T> *destination, Operand<T> *source) : m_destination(destination), m_source(source) {}

	std::string OpCode() const
	{
		return "abs" + T::Name();
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
class AbsoluteInstruction<Float32Type> : public PredicatedInstruction
{
public:
	AbsoluteInstruction(Register<Float32Type> *destination, Operand<Float32Type> *source) : m_destination(destination), m_source(source) {}

	void SetFlushSubNormal(bool flush) { m_flush = flush; }

	std::string OpCode() const
	{
		if (m_flush)
		{
			return "abs.ftz" + Float32Type::Name();
		}
		return "abs" + Float32Type::Name();
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
