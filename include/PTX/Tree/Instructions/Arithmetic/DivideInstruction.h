#pragma once

#include "PTX/Tree/Instructions/InstructionBase.h"
#include "PTX/Tree/Instructions/Modifiers/RoundingModifier.h"
#include "PTX/Tree/Instructions/Modifiers/FlushSubnormalModifier.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface(DivideInstruction)

template<class T, bool Assert = true>
class DivideInstruction : DispatchInherit(DivideInstruction), public InstructionBase_2<T>
{
public:
	REQUIRE_TYPE_PARAM(DivideInstruction,
		REQUIRE_EXACT(T,
			Int16Type, Int32Type, Int64Type,
			UInt16Type, UInt32Type, UInt64Type,
			Float32Type, Float64Type
		)
	);

	using InstructionBase_2<T>::InstructionBase_2;

	static std::string Mnemonic() { return "div"; }

	std::string OpCode() const override
	{
		return Mnemonic() + T::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);
};

template<>
class DivideInstruction<Float32Type> : DispatchInherit(DivideInstruction), public InstructionBase_2<Float32Type>, public RoundingModifier<Float32Type>, public FlushSubnormalModifier<Float32Type>
{
public:
	DivideInstruction(const Register<Float32Type> *destination, const TypedOperand<Float32Type> *sourceA, const TypedOperand<Float32Type> *sourceB, Float32Type::RoundingMode roundingMode = Float32Type::RoundingMode::None) : InstructionBase_2<Float32Type>(destination, sourceA, sourceB), RoundingModifier<Float32Type>(roundingMode) {}

	void SetRoundingMode(Float32Type::RoundingMode roundingMode)
	{
		if (roundingMode != Float32Type::RoundingMode::None)
		{
			m_full = false;
		}
		m_roundingMode = roundingMode;
	}

	bool GetFull() const { return m_full; }
	void SetFull(bool full)
	{
		if (full)
		{
			m_roundingMode == Float32Type::RoundingMode::None;
		}
		m_full = full;
	}

	static std::string Mnemonic() { return "div"; }

	std::string OpCode() const override
	{
		std::string code = Mnemonic();
		if (RoundingModifier<Float32Type>::IsActive())
		{
			code += RoundingModifier<Float32Type>::OpCodeModifier();
		}
		else if (m_full)
		{
			code += ".full";
		}
		else
		{
			code += ".approx";
		}
		return code + FlushSubnormalModifier<Float32Type>::OpCodeModifier() + Float32Type::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(Float32Type);

	bool m_full = false;
};

template<>
class DivideInstruction<Float64Type> : DispatchInherit(DivideInstruction), public InstructionBase_2<Float64Type>, public RoundingModifier<Float64Type, true>
{
public:
	DivideInstruction(const Register<Float64Type> *destination, const TypedOperand<Float64Type> *sourceA, const TypedOperand<Float64Type> *sourceB, Float64Type::RoundingMode roundingMode) : InstructionBase_2<Float64Type>(destination, sourceA, sourceB), RoundingModifier<Float64Type, true>(roundingMode) {}

	static std::string Mnemonic() { return "div"; }

	std::string OpCode() const override
	{
		return Mnemonic() + RoundingModifier<Float64Type, true>::OpCodeModifier() + Float64Type::Name();
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(Float64Type);
};

DispatchImplementation(DivideInstruction)
 
}
