#pragma once

#include <string>

#include "PTX/Operands/Operand.h"
#include "PTX/StateSpaces/StateSpace.h"
#include "PTX/StateSpaces/RegisterSpace.h"

namespace PTX {

template<Bits B, VectorSize V>
class UnsignedAdapter;

template<Bits B, VectorSize V>
class SignedAdapter;

template<Bits A>
class AddressRegister;

template<class T, VectorSize V = Scalar>
class Register : public Operand<T, V>
{
	static_assert(std::is_base_of<Type, T>::value, "T must be a PTX::Type");
public:
	Register(typename RegisterSpace<T, V>::Element *element, unsigned int index = 0) : m_structure(element), m_index(index) {}

	virtual std::string GetName() const
	{
		return m_structure->GetName(m_index);
	}

	std::string ToString() const
	{
		return GetName();
	}

	friend class UnsignedAdapter<Bits::Bits32, V>;
	friend class UnsignedAdapter<Bits::Bits64, V>;
	friend class SignedAdapter<Bits::Bits32, V>;
	friend class SignedAdapter<Bits::Bits64, V>;

	friend class RegisterSpace<T, V>;
	friend class IndexedRegister<T, VectorSize::Vector2>;
	friend class IndexedRegister<T, VectorSize::Vector4>;

	friend class AddressRegister<Bits::Bits32>;
	friend class AddressRegister<Bits::Bits64>;

protected:
	Register(Structure *structure, unsigned int index = 0) : m_structure(structure), m_index(index) {}

	Structure *m_structure = nullptr;
	unsigned int m_index;
};

}
