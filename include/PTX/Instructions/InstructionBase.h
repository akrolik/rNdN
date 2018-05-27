#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Type.h"
#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class D, class S = D>
class InstructionBase_1 : public PredicatedInstruction
{
public:
	InstructionBase_1(Register<D> *destination, Operand<S> *source) : m_destination(destination), m_source(source) {}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_source->ToString();
	}

private:
	Register<D> *m_destination = nullptr;
	Operand<S> *m_source = nullptr;
};

template<class D, class S1 = D, class S2 = S1>
class InstructionBase_2 : public PredicatedInstruction
{
public:
	InstructionBase_2(Register<D> *destination, Operand<S1> *sourceA, Operand<S2> *sourceB) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB) {}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString();
	}

private:
	Register<D> *m_destination = nullptr;
	Operand<S1> *m_sourceA = nullptr;
	Operand<S2> *m_sourceB = nullptr;
};

template<class D, class S1 = D, class S2 = S1, class S3 = S1>
class InstructionBase_3 : public PredicatedInstruction
{
public:
	InstructionBase_3(Register<D> *destination, Operand<S1> *sourceA, Operand<S2> *sourceB, Operand<S3> *sourceC) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC) {}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString() + ", " + m_sourceC->ToString();
	}

private:
	Register<D> *m_destination = nullptr;
	Operand<S1> *m_sourceA = nullptr;
	Operand<S2> *m_sourceB = nullptr;
	Operand<S3> *m_sourceC = nullptr;
};

template<class D, class S1 = D, class S2 = S1, class S3 = S1, class S4 = S1>
class InstructionBase_4 : public PredicatedInstruction
{
public:
	InstructionBase_4(Register<D> *destination, Operand<S1> *sourceA, Operand<S2> *sourceB, Operand<S3> *sourceC, Operand<S4> *sourceD) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_sourceD(sourceD) {}

	std::string Operands() const
	{
		return m_destination->ToString() + ", " + m_sourceA->ToString() + ", " + m_sourceB->ToString() + ", " + m_sourceC->ToString() + ", " + m_sourceD->ToString();
	}

private:
	Register<D> *m_destination = nullptr;
	Operand<S1> *m_sourceA = nullptr;
	Operand<S2> *m_sourceB = nullptr;
	Operand<S3> *m_sourceC = nullptr;
	Operand<S4> *m_sourceD = nullptr;
};

}
