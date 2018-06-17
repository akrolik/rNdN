#pragma once

#include <sstream>

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class D, class T, bool Typecheck = true>
class SetInstruction : public PredicatedInstruction, public FlushSubnormalModifier<T>
{
	//TODO: move this
	static_assert(
		std::is_same<D, Int32Type>::value ||
		std::is_same<D, UInt32Type>::value ||
		std::is_same<D, Float32Type>::value,
		"PTX::SetInstruction requires a typed 32-bit value"
	);
public:
	REQUIRE_TYPE(SetInstruction,
		Bit16Type, Bit32Type, Bit64Type,
		Int16Type, Int32Type, Int64Type,
		UInt16Type, UInt32Type, UInt64Type,
		Float16Type, Float16x2Type, Float32Type, Float64Type
	);

	enum BoolOperator {
		And,
		Or,
		Xor
	};

	SetInstruction(const Register<D> *destination, const Operand<T> *sourceA, const Operand<T> *sourceB, typename T::ComparisonOperator comparator) : SetInstruction(destination, sourceA, sourceB, comparator, nullptr, And, false) {}

	SetInstruction(const Register<D> *destination, const Operand<T> *sourceA, const Operand<T> *sourceB, typename T::ComparisonOperator comparator, const Register<PredicateType> *sourceC, BoolOperator boolOperator, bool negateSourcePredicate = false) : m_destination(destination), m_sourceA(sourceA), m_sourceB(sourceB), m_comparator(comparator), m_sourceC(sourceC), m_boolOperator(boolOperator), m_negateSourcePredicate(negateSourcePredicate) {}

	std::string OpCode() const override
	{
		std::ostringstream code;
		code << "set" << T::ComparisonOperatorString(m_comparator);
		if (m_sourceC != nullptr)
		{
			switch (m_boolOperator)
			{
				case And:
					code << ".and";
					break;
				case Or:
					code << ".or";
					break;
				case Xor:
					code << ".xor";
					break;
			}
		}
		if constexpr(T::FlushModifier)
		{
			if (this->m_flush)
			{
				code << ".ftz";
			}
		}
		code << D::Name() << T::Name();
		return code.str();
	}

	std::string Operands() const override
	{
		std::ostringstream code;
		code << m_destination->ToString() << ", " << m_sourceA->ToString() << ", " << m_sourceB->ToString();
		if (m_sourceC != nullptr)
		{
			code << ", ";
			if (m_negateSourcePredicate)
			{
				code << "!";
			}
			code << m_sourceC->ToString();
		}
		return code.str();
	}

private:
	const Register<D> *m_destination = nullptr;
	const Operand<T> *m_sourceA = nullptr;
	const Operand<T> *m_sourceB = nullptr;
	typename T::ComparisonOperator m_comparator;

	const Register<PredicateType> *m_sourceC = nullptr;
	BoolOperator m_boolOperator = And;
	bool m_negateSourcePredicate = false;
};

}
