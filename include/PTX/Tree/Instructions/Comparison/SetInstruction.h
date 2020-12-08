#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Tree/Instructions/Modifiers/PredicateModifier.h"

#include "PTX/Tree/Operands/Operand.h"
#include "PTX/Tree/Operands/Variables/Register.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface_2(SetInstruction)

template<class D, class T, bool Assert = true>
class SetInstruction : DispatchInherit(SetInstruction), public InstructionBase_2<D, T>, public FlushSubnormalModifier<T>, public PredicateModifier
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

	typename T::ComparisonOperator GetComparisonOperator() const { return m_comparator; }
	void SetComparisonOperator(typename T::ComparionOperator comparisonOperator) { m_comparator = comparisonOperator; }

	static std::string Mnemonic() { return "set"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic() + T::ComparisonOperatorString(m_comparator) + PredicateModifier::OpCodeModifier();
		if constexpr(FlushSubnormalModifier<T>::Enabled)
		{
			code += FlushSubnormalModifier<T>::OpCodeModifier();
		}
		return code + D::Name() + T::Name();
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

	// Visitors
	
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type1(D);
	DispatchMember_Type2(T);

	typename T::ComparisonOperator m_comparator;
};

DispatchImplementation_2(SetInstruction)

}
