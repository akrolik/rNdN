#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Type.h"
#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T, unsigned int N, class D = T>
class InstructionBase : public PredicatedInstruction
{
	REQUIRE_BASE_TYPE(InstructionBase, Type);
};

template<class T, class D>
class InstructionBase<T, 1, D> : public PredicatedInstruction
{
	REQUIRE_BASE_TYPE(InstructionBase, Type);
public:
	InstructionBase(Register<D> *destination, Operand<T> *source) : m_destination(destination), m_source(source) {}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}

private:
	Register<D> *m_destination = nullptr;
	Operand<T> *m_source = nullptr;
};

template<class T, class D>
class InstructionBase<T, 2, D> : public PredicatedInstruction
{
	REQUIRE_BASE_TYPE(InstructionBase, Type);
public:
	InstructionBase(Register<D> *destination, Operand<T> *sourceA, Operand<T> *sourceB) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB) {}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString();
	}

private:
	Register<D> *m_destination = nullptr;
	Operand<T> *m_sourceA = nullptr;
	Operand<T> *m_sourceB = nullptr;
};

template<class T, class D>
class InstructionBase<T, 3, D> : public PredicatedInstruction
{
	REQUIRE_BASE_TYPE(InstructionBase, Type);
public:
	InstructionBase(Register<D> *destination, Operand<T> *sourceA, Operand<T> *sourceB, Operand<T> *sourceC) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC) {}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString() + ", " + m_sourceC->ToString();
	}

private:
	Register<D> *m_destination = nullptr;
	Operand<T> *m_sourceA = nullptr;
	Operand<T> *m_sourceB = nullptr;
	Operand<T> *m_sourceC = nullptr;
};

}
