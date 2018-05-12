#pragma once

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/VariableSet.h"
#include "PTX/StateSpaces/StateSpace.h"
#include "PTX/StateSpaces/SpaceAdapter.h"

namespace PTX {

template<class T, class S>
class Variable : public Operand<T>, public VariableSet<T, S>
{
	friend class VariableSet<T, S>;
	friend S;

	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
	static_assert(std::is_base_of<StateSpace<T>, S>::value, "S must be a PTX::StateSpace<T>");
public:
	Variable() {}

	virtual std::string GetName() const
	{
		return m_name;
	}

	Variable<T, S> *GetVariable(unsigned int index) const
	{
		if (m_count == 1 && index == 0)
		{
			return this;
		}
		return VariableSet<T, S>::GetVariable(index);
	}

	std::string ToString() const
	{
		return VariableSet<T, S>::ToString();
	}

protected:
	Variable(std::string name, S *stateSpace) : VariableSet<T, S>(name, 1, stateSpace) {}

	using VariableSet<T, S>::m_count;
	using VariableSet<T, S>::m_name;
};

template<class T>
class RegisterSpace;
template<class T>
using Register = Variable<T, RegisterSpace<T>>;

template<class T>
class MemorySpace;
template<class T>
using MemoryVariable = Variable<T, MemorySpace<T>>;

template<class T, VectorSize V>
class IndexedRegister : public Register<T>
{
public:
	IndexedRegister(Register<VectorType<T, V>> *variable, VectorElement vectorElement) : Register<T>(variable->GetName(), new RegisterSpaceAdapter<VectorType<T, V>, T>(variable->GetStateSpace())), m_vectorElement(vectorElement) {}

	virtual VectorElement GetVectorElement() const { return m_vectorElement; }

	std::string ToString() const
	{
		return Register<T>::ToString() + GetVectorElementName(m_vectorElement);
	}

private:
	VectorElement m_vectorElement;
};

template<Bits A, class T>
class AddressRegister : public Register<UIntType<A>>
{
public:
	AddressRegister(Register<UIntType<A>> *variable, MemorySpace<T> *memorySpace, AddressSpace addressSpace = AddressSpace::Generic) : m_variable(variable), m_memorySpace(memorySpace), m_addressSpace(addressSpace) {}

	virtual AddressSpace GetAddressSpace() const { return m_addressSpace; }

	std::string ToString() const
	{
		return m_variable->ToString();
	}

private:
	Register<UIntType<A>> *m_variable;
	MemorySpace<T> *m_memorySpace;
	AddressSpace m_addressSpace;
};

template<class T>
using Address32Register = AddressRegister<Bits::Bits32, T>;
template<class T>
using Address64Register = AddressRegister<Bits::Bits64, T>;

}
