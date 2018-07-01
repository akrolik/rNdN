#pragma once

#include <sstream>

#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/PredicateModifier.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class D, class T, bool Assert = true>
class SetInstruction : public InstructionBase_2<D, T>, public FlushSubnormalModifier<T>, public PredicateModifier
{
public:
	REQUIRE_TYPE_PARAMS(SetInstruction,
		REQUIRE_EXACT(D,
			Int32Type, UInt32Type, Float32Type
		),
		REQUIRE_EXACT(T,
			Bit16Type, Bit32Type, Bit64Type,
			Int16Type, Int32Type, Int64Type,
			UInt16Type, UInt32Type, UInt64Type,
			Float16Type, Float16x2Type, Float32Type, Float64Type
		)
	);

	SetInstruction(const Register<D> *destination, const TypedOperand<T> *sourceA, const TypedOperand<T> *sourceB, typename T::ComparisonOperator comparator) : InstructionBase_2<D, T>(destination, sourceA, sourceB), m_comparator(comparator) {}

	SetInstruction(const Register<D> *destination, const TypedOperand<T> *sourceA, const TypedOperand<T> *sourceB, typename T::ComparisonOperator comparator, const Register<PredicateType> *sourceC, BoolOperator boolOperator, bool negateSourcePredicate = false) : InstructionBase_2<D, T>(destination, sourceA, sourceB), m_comparator(comparator), PredicateModifier(sourceC, boolOperator, negateSourcePredicate) {}

	std::string OpCode() const override
	{
		std::ostringstream code;
		code << "set" << T::ComparisonOperatorString(m_comparator) << PredicateModifier::OpCodeModifier();
		if constexpr(FlushSubnormalModifier<T>::Enabled)
		{
			code << FlushSubnormalModifier<T>::OpCodeModifier();
		}
		code << D::Name() << T::Name();
		return code.str();
	}

	std::vector<const Operand *> Operands() const override
	{
		auto operands = InstructionBase_2<D, T>::Operands();
		const Operand *modifier = PredicateModifier::OperandsModifier();
		if (modifier != nullptr)
		{
			operands.push_back(modifier);
		}
		return operands;
	}

private:
	typename T::ComparisonOperator m_comparator;
};

}
