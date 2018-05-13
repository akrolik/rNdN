#pragma once

#include "PTX/Operands/Operand.h"

namespace PTX {

template<class T>
class StateSpace;

template<class T, class S>
class Variable : public Operand<T>
{
	friend S;

	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
	static_assert(std::is_base_of<StateSpace<T>, S>::value, "S must be a PTX::StateSpace<T>");
public:
	virtual std::string GetName() const
	{
		return m_name;
	}

	virtual S *GetStateSpace() const
	{
		return m_stateSpace;
	}

	std::string ToString() const
	{
		return GetName();
	}

protected:
	Variable(std::string name, S *stateSpace) : m_name(name), m_stateSpace(stateSpace) {}

	std::string m_name;
	S *m_stateSpace = nullptr;
};

#include "PTX/StateSpaces/SpaceAdapter.h"

template<class T, VectorSize V>
class IndexedRegister : public Register<T>
{
public:
	IndexedRegister(Register<VectorType<T, V>> *variable, VectorElement vectorElement) : Register<T>(variable->GetName(), new RegisterSpaceAdapter<T, VectorType<T, V>>(variable->GetStateSpace())), m_vectorElement(vectorElement) {}

	virtual VectorElement GetVectorElement() const { return m_vectorElement; }

	std::string ToString() const
	{
		return Register<T>::ToString() + GetVectorElementName(m_vectorElement);
	}

private:
	VectorElement m_vectorElement;
};

template<Bits A, class T, AddressSpace S>
class AddressRegister// : public Register<UIntType<A>>
{
public:
	AddressRegister(Register<UIntType<A>> *variable, AddressableSpace<T, S> *memorySpace) : m_variable(variable), m_memorySpace(memorySpace) {}

	std::string ToString() const
	{
		return m_variable->ToString();
	}

private:
	Register<UIntType<A>> *m_variable;
	AddressableSpace<T, S> *m_memorySpace;
};

template<class T, AddressSpace S>
using Address32Register = AddressRegister<Bits::Bits32, T, S>;
template<class T, AddressSpace S>
using Address64Register = AddressRegister<Bits::Bits64, T, S>;

}
