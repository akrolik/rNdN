#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Tree/Instructions/Modifiers/PredicateModifier.h"

#include "PTX/Tree/Operands/Operand.h"
#include "PTX/Tree/Operands/Variables/Registers/Register.h"

namespace PTX {

DispatchInterface_2(SetInstruction)

template<class D, class T, bool Assert = true>
class SetInstruction : DispatchInherit(SetInstruction), public InstructionBase_2<D, T>, public FlushSubnormalModifier<T>, public PredicateModifier
{
public:
	REQUIRE_TYPE_PARAM(SetInstruction,
		(
			REQUIRE_EXACT(D,
				Int32Type, UInt32Type, Float16Type, Float32Type
			) &&
			REQUIRE_EXACT(T,
				Bit16Type, Bit32Type, Bit64Type,
				Int16Type, Int32Type, Int64Type,
				UInt16Type, UInt32Type, UInt64Type,
				Float32Type, Float64Type
			)
		) || (
			REQUIRE_EXACT(D,
                        	Float16Type, UInt16Type, Int16Type, UInt32Type, Int32Type
			) &&
			REQUIRE_EXACT(T, Float16Type)
		) || (
			REQUIRE_EXACT(D,
                        	Float16x2Type, UInt32Type, Int32Type
			) &&
			REQUIRE_EXACT(T, Float16x2Type)
		)
	);

	SetInstruction(Register<D> *destination, TypedOperand<T> *sourceA, TypedOperand<T> *sourceB, typename T::ComparisonOperator comparator)
		: InstructionBase_2<D, T>(destination, sourceA, sourceB), m_comparator(comparator) {}

	SetInstruction(Register<D> *destination, TypedOperand<T> *sourceA, TypedOperand<T> *sourceB, typename T::ComparisonOperator comparator, Register<PredicateType> *sourceC, BoolOperator boolOperator, bool negateSourcePredicate = false)
		: InstructionBase_2<D, T>(destination, sourceA, sourceB), m_comparator(comparator), PredicateModifier(sourceC, boolOperator, negateSourcePredicate) {}

	// Analysis properties

	bool HasSideEffect() const override { return false; }

	// Properties

	typename T::ComparisonOperator GetComparisonOperator() const { return m_comparator; }
	void SetComparisonOperator(typename T::ComparionOperator comparisonOperator) { m_comparator = comparisonOperator; }

	// Formatting

	static std::string Mnemonic() { return "set"; }

	std::string GetOpCode() const override
	{
		std::string code = Mnemonic() + T::ComparisonOperatorString(m_comparator) + PredicateModifier::GetOpCodeModifier();
		if constexpr(FlushSubnormalModifier<T>::Enabled)
		{
			code += FlushSubnormalModifier<T>::GetOpCodeModifier();
		}
		return code + D::Name() + T::Name();
	}

	std::vector<const Operand *> GetOperands() const override
	{
		auto operands = InstructionBase_2<D, T>::GetOperands();
		if (const auto modifier = PredicateModifier::GetOperandsModifier())
		{
			operands.push_back(modifier);
		}
		return operands;
	}

	std::vector<Operand *> GetOperands() override
	{
		auto operands = InstructionBase_2<D, T>::GetOperands();
		if (auto modifier = PredicateModifier::GetOperandsModifier())
		{
			operands.push_back(modifier);
		}
		return operands;
	}

	// Visitors
	
	void Accept(InstructionVisitor& visitor) override { visitor.Visit(this); }
	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type1(D);
	DispatchMember_Type2(T);

	typename T::ComparisonOperator m_comparator;
};

}
