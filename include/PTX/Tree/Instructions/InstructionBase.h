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
	InstructionBase_0(Register<D> *destination) : m_destination(destination) {}
	 
	// Properties

	const Register<D> *GetDestination() const { return m_destination; }
	Register<D> *GetDestination() { return m_destination; }
	void SetDestination(Register<D> *destination) { m_destination = destination; }

	// Formatting

	std::vector<const Operand *> GetOperands() const override
	{
		return { m_destination };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { m_destination };
	}

protected:
	Register<D> *m_destination = nullptr;
};

template<class D, class S = D>
class InstructionBase_1 : public PredicatedInstruction
{
public:
	InstructionBase_1(Register<D> *destination, TypedOperand<S> *source) : m_destination(destination), m_source(source) {}

	// Properties

	const Register<D> *GetDestination() const { return m_destination; }
	Register<D> *GetDestination() { return m_destination; }
	void SetDestination(Register<D> *destination) { m_destination = destination; }

	const TypedOperand<S> *GetSource() const { return m_source; }
	TypedOperand<S> *GetSource() { return m_source; }
	void SetSource(TypedOperand<S> *source) { m_source = source; }

	// Formatting

	std::vector<const Operand *> GetOperands() const override
	{
		return { m_destination, m_source };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { m_destination, m_source };
	}

protected:
	Register<D> *m_destination = nullptr;
	TypedOperand<S> *m_source = nullptr;
};

template<class D, class S1 = D, class S2 = S1>
class InstructionBase_2 : public PredicatedInstruction
{
public:
	InstructionBase_2(Register<D> *destination, TypedOperand<S1> *sourceA, TypedOperand<S2> *sourceB) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB) {}

	// Properties

	const Register<D> *GetDestination() const { return m_destination; }
	Register<D> *GetDestination() { return m_destination; }
	void SetDestination(Register<D> *destination) { m_destination = destination; }

	const TypedOperand<S1> *GetSourceA() const { return m_sourceA; }
	TypedOperand<S1> *GetSourceA() { return m_sourceA; }
	void SetSourceA(TypedOperand<S1> *source) { m_sourceA = source; }

	const TypedOperand<S2> *GetSourceB() const { return m_sourceB; }
	TypedOperand<S2> *GetSourceB() { return m_sourceB; }
	void SetSourceB(TypedOperand<S2> *source) { m_sourceB = source; }

	// Formatting

	std::vector<const Operand *> GetOperands() const override
	{
		return { m_destination, m_sourceA, m_sourceB };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { m_destination, m_sourceA, m_sourceB };
	}

protected:
	Register<D> *m_destination = nullptr;
	TypedOperand<S1> *m_sourceA = nullptr;
	TypedOperand<S2> *m_sourceB = nullptr;
};

template<class D, class S1 = D, class S2 = S1, class S3 = S1>
class InstructionBase_3 : public PredicatedInstruction
{
public:
	InstructionBase_3(Register<D> *destination, TypedOperand<S1> *sourceA, TypedOperand<S2> *sourceB, TypedOperand<S3> *sourceC) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC) {}

	// Properties

	const Register<D> *GetDestination() const { return m_destination; }
	Register<D> *GetDestination() { return m_destination; }
	void SetDestination(Register<D> *destination) { m_destination = destination; }

	const TypedOperand<S1> *GetSourceA() const { return m_sourceA; }
	TypedOperand<S1> *GetSourceA() { return m_sourceA; }
	void SetSourceA(TypedOperand<S1> *source) { m_sourceA = source; }

	const TypedOperand<S2> *GetSourceB() const { return m_sourceB; }
	TypedOperand<S2> *GetSourceB() { return m_sourceB; }
	void SetSourceB(TypedOperand<S2> *source) { m_sourceB = source; }

	const TypedOperand<S3> *GetSourceC() const { return m_sourceC; }
	TypedOperand<S3> *GetSourceC() { return m_sourceC; }
	void SetSourceC(TypedOperand<S3> *source) { m_sourceC = source; }

	// Formatting

	std::vector<const Operand *> GetOperands() const override
	{
		return { m_destination, m_sourceA, m_sourceB, m_sourceC };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { m_destination, m_sourceA, m_sourceB, m_sourceC };
	}

protected:
	Register<D> *m_destination = nullptr;
	TypedOperand<S1> *m_sourceA = nullptr;
	TypedOperand<S2> *m_sourceB = nullptr;
	TypedOperand<S3> *m_sourceC = nullptr;
};

template<class D, class S1 = D, class S2 = S1, class S3 = S1, class S4 = S1>
class InstructionBase_4 : public PredicatedInstruction
{
public:
	InstructionBase_4(Register<D> *destination, TypedOperand<S1> *sourceA, TypedOperand<S2> *sourceB, TypedOperand<S3> *sourceC, TypedOperand<S4> *sourceD) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_sourceD(sourceD) {}
	
	// Formatting

	const Register<D> *GetDestination() const { return m_destination; }
	Register<D> *GetDestination() { return m_destination; }
	void SetDestination(Register<D> *destination) { m_destination = destination; }

	const TypedOperand<S1> *GetSourceA() const { return m_sourceA; }
	TypedOperand<S1> *GetSourceA() { return m_sourceA; }
	void SetSourceA(TypedOperand<S1> *source) { m_sourceA = source; }

	const TypedOperand<S2> *GetSourceB() const { return m_sourceB; }
	TypedOperand<S2> *GetSourceB() { return m_sourceB; }
	void SetSourceB(TypedOperand<S2> *source) { m_sourceB = source; }

	const TypedOperand<S3> *GetSourceC() const { return m_sourceC; }
	TypedOperand<S3> *GetSourceC() { return m_sourceC; }
	void SetSourceC(TypedOperand<S3> *source) { m_sourceC = source; }

	const TypedOperand<S4> *GetSourceD() const { return m_sourceD; }
	TypedOperand<S4> *GetSourceD() { return m_sourceD; }
	void SetSourceD(TypedOperand<S4> *source) { m_sourceD = source; }

	// Formatting

	std::vector<const Operand *> GetOperands() const override
	{
		return { m_destination, m_sourceA, m_sourceB, m_sourceC, m_sourceD };
	}

	std::vector<Operand *> GetOperands() override
	{
		return { m_destination, m_sourceA, m_sourceB, m_sourceC, m_sourceD };
	}

protected:
	Register<D> *m_destination = nullptr;
	TypedOperand<S1> *m_sourceA = nullptr;
	TypedOperand<S2> *m_sourceB = nullptr;
	TypedOperand<S3> *m_sourceC = nullptr;
	TypedOperand<S4> *m_sourceD = nullptr;
};

}
