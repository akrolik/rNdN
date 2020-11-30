#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Type.h"
#include "PTX/Tree/Operands/Operand.h"
#include "PTX/Tree/Operands/Variables/Register.h"

namespace PTX {

template<class D>
class InstructionBase_0 : public PredicatedInstruction
{
public:
	InstructionBase_0(const Register<D> *destination) : m_destination(destination) {}

	const Register<D> *GetDestination() const { return m_destination; }
	void SetDestination(const Register<D> *destination) { m_destination = destination; }

	std::vector<const Operand *> Operands() const override
	{
		return { m_destination };
	}

private:
	const Register<D> *m_destination = nullptr;
};

template<class D, class S = D>
class InstructionBase_1 : public PredicatedInstruction
{
public:
	InstructionBase_1(const Register<D> *destination, const TypedOperand<S> *source) : m_destination(destination), m_source(source) {}

	const Register<D> *GetDestination() const { return m_destination; }
	void SetDestination(const Register<D> *destination) { m_destination = destination; }

	const TypedOperand<S> *GetSource() const { return m_source; }
	void SetSource(const TypedOperand<S> *source) { m_source = source; }

	std::vector<const Operand *> Operands() const override
	{
		return { m_destination, m_source };
	}

private:
	const Register<D> *m_destination = nullptr;
	const TypedOperand<S> *m_source = nullptr;
};

template<class D, class S1 = D, class S2 = S1>
class InstructionBase_2 : public PredicatedInstruction
{
public:
	InstructionBase_2(const Register<D> *destination, const TypedOperand<S1> *sourceA, const TypedOperand<S2> *sourceB) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB) {}

	const Register<D> *GetDestination() const { return m_destination; }
	void SetDestination(const Register<D> *destination) { m_destination = destination; }

	const TypedOperand<S1> *GetSourceA() const { return m_sourceA; }
	void SetSourceA(const TypedOperand<S1> *source) { m_sourceA = source; }

	const TypedOperand<S2> *GetSourceB() const { return m_sourceB; }
	void SetSourceB(const TypedOperand<S2> *source) { m_sourceB = source; }

	std::vector<const Operand *> Operands() const override
	{
		return { m_destination, m_sourceA, m_sourceB };
	}

private:
	const Register<D> *m_destination = nullptr;
	const TypedOperand<S1> *m_sourceA = nullptr;
	const TypedOperand<S2> *m_sourceB = nullptr;
};

template<class D, class S1 = D, class S2 = S1, class S3 = S1>
class InstructionBase_3 : public PredicatedInstruction
{
public:
	InstructionBase_3(const Register<D> *destination, const TypedOperand<S1> *sourceA, const TypedOperand<S2> *sourceB, const TypedOperand<S3> *sourceC) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC) {}

	const Register<D> *GetDestination() const { return m_destination; }
	void SetDestination(const Register<D> *destination) { m_destination = destination; }

	const TypedOperand<S1> *GetSourceA() const { return m_sourceA; }
	void SetSourceA(const TypedOperand<S1> *source) { m_sourceA = source; }

	const TypedOperand<S2> *GetSourceB() const { return m_sourceB; }
	void SetSourceB(const TypedOperand<S2> *source) { m_sourceB = source; }

	const TypedOperand<S3> *GetSourceC() const { return m_sourceC; }
	void SetSourceC(const TypedOperand<S3> *source) { m_sourceC = source; }

	std::vector<const Operand *> Operands() const override
	{
		return { m_destination, m_sourceA, m_sourceB, m_sourceC };
	}

private:
	const Register<D> *m_destination = nullptr;
	const TypedOperand<S1> *m_sourceA = nullptr;
	const TypedOperand<S2> *m_sourceB = nullptr;
	const TypedOperand<S3> *m_sourceC = nullptr;
};

template<class D, class S1 = D, class S2 = S1, class S3 = S1, class S4 = S1>
class InstructionBase_4 : public PredicatedInstruction
{
public:
	InstructionBase_4(const Register<D> *destination, const TypedOperand<S1> *sourceA, const TypedOperand<S2> *sourceB, const TypedOperand<S3> *sourceC, const TypedOperand<S4> *sourceD) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_sourceD(sourceD) {}

	const Register<D> *GetDestination() const { return m_destination; }
	void SetDestination(const Register<D> *destination) { m_destination = destination; }

	const TypedOperand<S1> *GetSourceA() const { return m_sourceA; }
	void SetSourceA(const TypedOperand<S1> *source) { m_sourceA = source; }

	const TypedOperand<S2> *GetSourceB() const { return m_sourceB; }
	void SetSourceB(const TypedOperand<S2> *source) { m_sourceB = source; }

	const TypedOperand<S3> *GetSourceC() const { return m_sourceC; }
	void SetSourceC(const TypedOperand<S3> *source) { m_sourceC = source; }

	const TypedOperand<S4> *GetSourceD() const { return m_sourceD; }
	void SetSourceD(const TypedOperand<S4> *source) { m_sourceD = source; }

	std::vector<const Operand *> Operands() const override
	{
		return { m_destination, m_sourceA, m_sourceB, m_sourceC, m_sourceD };
	}

private:
	const Register<D> *m_destination = nullptr;
	const TypedOperand<S1> *m_sourceA = nullptr;
	const TypedOperand<S2> *m_sourceB = nullptr;
	const TypedOperand<S3> *m_sourceC = nullptr;
	const TypedOperand<S4> *m_sourceD = nullptr;
};

}
