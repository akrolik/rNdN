#pragma once

#include "PTX/Instructions/InstructionBase.h"
#include "PTX/Instructions/PredicatedInstruction.h"
#include "PTX/Instructions/Modifiers/ComparisonModifier.h"
#include "PTX/Instructions/Modifiers/FlushSubnormalModifier.h"
#include "PTX/Instructions/Modifiers/PredicateModifier.h"

#include "PTX/Type.h"
#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Extended/DualOperand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T, bool Assert = true>
class SetPredicateInstruction : public PredicatedInstruction, public ComparisonModifier<T>, public FlushSubnormalModifier<T>, public PredicateModifier
{
public:
	REQUIRE_TYPE_PARAM(SetPredicateInstruction,
		REQUIRE_EXACT(T,
			Bit16Type, Bit32Type, Bit64Type,
			Int16Type, Int32Type, Int64Type,
			UInt16Type, UInt32Type, UInt64Type,
			Float16Type, Float16x2Type, Float32Type, Float64Type
		)
	);

	SetPredicateInstruction(const Register<PredicateType> *destination, const TypedOperand<T> *sourceA, const TypedOperand<T> *sourceB, typename T::ComparisonOperator comparator) : ComparisonModifier<T>(comparator), m_destinationP(destination), m_sourceA(sourceA), m_sourceB(sourceB) {}

	SetPredicateInstruction(const Register<PredicateType> *destinationP, const Register<PredicateType> *destinationQ, const TypedOperand<T> *sourceA, const TypedOperand<T> *sourceB, typename T::ComparisonOperator comparator, const Register<PredicateType> *sourceC, BoolOperator boolOperator, bool negateSourcePredicate = false) : ComparisonModifier<T>(comparator), m_destinationP(destinationP), m_destinationQ(destinationQ), m_sourceA(sourceA), m_sourceB(sourceB), PredicateModifier(sourceC, boolOperator, negateSourcePredicate) {}

	const Register<PredicateType> *GetDestination() const { return m_destinationP; }
	void SetDestination(const Register<PredicateType> *destination) { m_destinationP = destination; }

	const Register<PredicateType> *GetDestinationQ() const { return m_destinationQ; }
	void SetDestinationQ(const Register<PredicateType> *destination) { m_destinationQ = destination; }

	const TypedOperand<T> *GetSourceA() const { return m_sourceA; }
	void SetSourceA(const TypedOperand<T> *source) { m_sourceA = source; }

	const TypedOperand<T> *GetSourceB() const { return m_sourceB; }
	void SetSourceB(const TypedOperand<T> *source) { m_sourceB = source; }

	static std::string Mnemonic() { return "setp"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic() + ComparisonModifier<T>::OpCodeModifier() + PredicateModifier::OpCodeModifier();
		if constexpr(FlushSubnormalModifier<T>::Enabled)
		{
			code += FlushSubnormalModifier<T>::OpCodeModifier();
		}
		return code + T::Name();
	}

	std::vector<const Operand *> Operands() const override
	{
		std::vector<const Operand *> operands;
		if (m_destinationQ == nullptr)
		{
			operands.push_back(m_destinationP);
		}
		else
		{
			operands.push_back(new DualOperand(m_destinationP, m_destinationQ));
		}
		operands.push_back(m_sourceA);
		operands.push_back(m_sourceB);
		const Operand *modifier = PredicateModifier::OperandsModifier();
		if (modifier != nullptr)
		{
			operands.push_back(modifier);
		}
		return operands;
	}

private:
	const Register<PredicateType> *m_destinationP = nullptr;
	const Register<PredicateType> *m_destinationQ = nullptr;
	const TypedOperand<T> *m_sourceA = nullptr;
	const TypedOperand<T> *m_sourceB = nullptr;
};

template<>
class SetPredicateInstruction<Float16Type> : public InstructionBase_2<PredicateType, Float16Type>, public ComparisonModifier<Float16Type>, public FlushSubnormalModifier<Float16Type>, public PredicateModifier
{
public:
	SetPredicateInstruction(const Register<PredicateType> *destination, const TypedOperand<Float16Type> *sourceA, const TypedOperand<Float16Type> *sourceB, Float16Type::ComparisonOperator comparator) : InstructionBase_2<PredicateType, Float16Type>(destination, sourceA, sourceB), ComparisonModifier<Float16Type>(comparator) {}

	SetPredicateInstruction(const Register<PredicateType> *destination, const TypedOperand<Float16Type> *sourceA, const TypedOperand<Float16Type> *sourceB, Float16Type::ComparisonOperator comparator, const Register<PredicateType> *sourceC, BoolOperator boolOperator, bool negateSourcePredicate = false) : InstructionBase_2<PredicateType, Float16Type>(destination, sourceA, sourceB), ComparisonModifier<Float16Type>(comparator), PredicateModifier(sourceC, boolOperator, negateSourcePredicate) {}

	static std::string Mnemonic() { return "setp"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ComparisonModifier<Float16Type>::OpCodeModifier() + PredicateModifier::OpCodeModifier() + FlushSubnormalModifier<Float16Type>::OpCodeModifier() + Float16Type::Name();
	}

	std::vector<const Operand *> Operands() const override
	{
		auto operands = InstructionBase_2<PredicateType, Float16Type>::Operands();
		const Operand *modifier = PredicateModifier::OperandsModifier();
		if (modifier != nullptr)
		{
			operands.push_back(modifier);
		}
		return operands;
	}
};

}
