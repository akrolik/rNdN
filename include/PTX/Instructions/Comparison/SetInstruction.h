#pragma once

#include <sstream>

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/PredicateModifier.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class D, class T, bool Typecheck = true>
class SetInstruction : public InstructionBase_2<D, T>, public FlushSubnormalModifier<T>, public PredicateModifier
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

	SetInstruction(const Register<D> *destination, const Operand<T> *sourceA, const Operand<T> *sourceB, typename T::ComparisonOperator comparator) : InstructionBase_2<D, T>(destination, sourceA, sourceB), m_comparator(comparator) {}

	SetInstruction(const Register<D> *destination, const Operand<T> *sourceA, const Operand<T> *sourceB, typename T::ComparisonOperator comparator, const Register<PredicateType> *sourceC, BoolOperator boolOperator, bool negateSourcePredicate = false) : InstructionBase_2<D, T>(destination, sourceA, sourceB), m_comparator(comparator), PredicateModifier(sourceC, boolOperator, negateSourcePredicate) {}

	std::string OpCode() const override
	{
		std::ostringstream code;
		code << "set" << T::ComparisonOperatorString(m_comparator) << PredicateModifier::OpCodeModifier();
		if constexpr(T::FlushModifier)
		{
			code << FlushSubnormalModifier<T>::OpCodeModifier();
		}
		code << D::Name() << T::Name();
		return code.str();
	}

	std::string Operands() const override
	{
		return InstructionBase_2<D, T>::Operands() + PredicateModifier::OperandsModifier();
	}

private:
	typename T::ComparisonOperator m_comparator;
};

}
