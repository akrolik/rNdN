#pragma once

#include "PTX/Tree/Type.h"
#include "PTX/Tree/Node.h"

namespace PTX {

class Operand : public Node
{
public:
	std::string ToString(unsigned int indentation) const override
	{
		return ToString();
	}

	virtual std::string ToString() const = 0;

	// Visitors

	void Accept(ConstHierarchicalVisitor& visitor) const override
	{
		visitor.VisitIn(this);
		visitor.VisitOut(this);
	}

	template<class V> void Dispatch(V& visitor) const;
	template<class V> bool DispatchIn(V& visitor) const;
	template<class V> void DispatchOut(V& visitor) const;

protected:
	//TODO:
	virtual const Type *GetType() const { return nullptr; }
};

template<class T>
class TypedOperand : public virtual Operand
{
	REQUIRE_TYPE_PARAM(Operand,
		REQUIRE_BASE(T, Type)
	);

protected:
	const T *GetType() const { return &m_type; }
	T m_type;
};

template<class V>
void Operand::Dispatch(V& visitor) const
{
}

template<class V>
bool Operand::DispatchIn(V& visitor) const
{
#define Operand_TypeDispatch(T) \
	if (dynamic_cast<const T*>(type)) { \
		return visitor.VisitIn(dynamic_cast<const TypedOperand<T>*>(this)); \
	}

	const auto type = GetType();
	if (type == nullptr)
	{
		//TODO:
	}

	// Int
	Operand_TypeDispatch(IntType<Bits::Bits8>);
	Operand_TypeDispatch(IntType<Bits::Bits16>);
	Operand_TypeDispatch(IntType<Bits::Bits32>);
	Operand_TypeDispatch(IntType<Bits::Bits64>);

	// UInt
	Operand_TypeDispatch(UIntType<Bits::Bits8>);
	Operand_TypeDispatch(UIntType<Bits::Bits16>);
	Operand_TypeDispatch(UIntType<Bits::Bits32>);
	Operand_TypeDispatch(UIntType<Bits::Bits64>);

	// Float
	Operand_TypeDispatch(FloatType<Bits::Bits16>);
	Operand_TypeDispatch(FloatType<Bits::Bits32>);
	Operand_TypeDispatch(FloatType<Bits::Bits64>);

	// Bit
	Operand_TypeDispatch(BitType<Bits::Bits1>);
	Operand_TypeDispatch(BitType<Bits::Bits8>);
	Operand_TypeDispatch(BitType<Bits::Bits16>);
	Operand_TypeDispatch(BitType<Bits::Bits32>);
	Operand_TypeDispatch(BitType<Bits::Bits64>);

	return true;
}

template<class V>
void Operand::DispatchOut(V& visitor) const
{
}

}
