#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"

#include "PTX/Tree/Operands/Operand.h"
#include "PTX/Tree/Operands/Extended/DualOperand.h"
#include "PTX/Tree/Operands/Extended/HexOperand.h"
#include "PTX/Tree/Operands/Variables/Register.h"

#include "PTX/Traversal/InstructionDispatch.h"

namespace PTX {

DispatchInterface(ShuffleInstruction)

template<class T>
class ShuffleInstruction : DispatchInherit(ShuffleInstruction), public PredicatedInstruction
{
public:
	REQUIRE_TYPE_PARAM(ShuffleInstruction,
		REQUIRE_EXACT(T, Bit32Type)
	);

	enum class Mode {
		Up,
		Down,
		Butterfly,
		Index
	};

	static std::string ModeString(Mode mode)
	{
		switch (mode)
		{
			case Mode::Up:
				return ".up";
			case Mode::Down:
				return ".down";
			case Mode::Butterfly:
				return ".bfly";
			case Mode::Index:
				return ".idx";
		}
		return ".<unknown>";
	}

	ShuffleInstruction(Register<T> *destinationD, TypedOperand<T> *sourceA, TypedOperand<UInt32Type> *sourceB, TypedOperand<UInt32Type> *sourceC, uint32_t memberMask, Mode mode)
		: m_destinationD(destinationD), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_memberMask(memberMask), m_mode(mode) {}
	ShuffleInstruction(Register<T> *destinationD, Register<PredicateType> *destinationP, TypedOperand<T> *sourceA, TypedOperand<UInt32Type> *sourceB, TypedOperand<UInt32Type> *sourceC, uint32_t memberMask, Mode mode)
		: m_destinationD(destinationD), m_destinationP(destinationP), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_memberMask(memberMask), m_mode(mode) {}

	// Properties

	const Register<T> *GetDestination() const { return m_destinationD; }
	Register<T> *GetDestination() { return m_destinationD; }
	void SetDestination(Register<T> *destination) { m_destinationD = destination; }

	const Register<PredicateType> *GetDestinationP() const { return m_destinationP; }
	Register<PredicateType> *GetDestinationP() { return m_destinationP; }
	void SetDestinationP(Register<PredicateType> *destination) { m_destinationP = destination; }

	const TypedOperand<T> *GetSourceA() const { return m_sourceA; }
	TypedOperand<T> *GetSourceA() { return m_sourceA; }
	void SetSourceA(TypedOperand<T> *source) { m_sourceA = source; }

	const TypedOperand<UInt32Type> *GetSourceB() const { return m_sourceB; }
	TypedOperand<UInt32Type> *GetSourceB() { return m_sourceB; }
	void SetSourceB(TypedOperand<UInt32Type> *source) { m_sourceB = source; }

	const TypedOperand<UInt32Type> *GetSourceC() const { return m_sourceC; }
	TypedOperand<UInt32Type> *GetSourceC() { return m_sourceC; }
	void SetSourceC(TypedOperand<UInt32Type> *source) { m_sourceC = source; }

	Mode GetMode() const { return m_mode; }
	void SetMode(Mode mode) { m_mode = mode; }

	uint32_t GetMemberMask() const { return m_memberMask; }
	void SetMemberMask(uint32_t memberMask) { m_memberMask = memberMask; }

	// Formatting

	static std::string Mnemonic() { return "shfl"; }

	std::string GetOpCode() const override
	{
		return Mnemonic() + ".sync" + ModeString(m_mode) + T::Name();
	}

	std::vector<const Operand *> GetOperands() const override
	{
		std::vector<const Operand *> operands;
		if (m_destinationP == nullptr)
		{
			operands.push_back(m_destinationD);
		}
		else
		{
			operands.push_back(new DualOperand(m_destinationD, m_destinationP));
		}
		operands.push_back(m_sourceA);
		operands.push_back(m_sourceB);
		operands.push_back(m_sourceC);
		operands.push_back(new HexOperand(m_memberMask));
		return operands;
	}

	std::vector<Operand *> GetOperands() override
	{
		std::vector<Operand *> operands;
		if (m_destinationP == nullptr)
		{
			operands.push_back(m_destinationD);
		}
		else
		{
			operands.push_back(new DualOperand(m_destinationD, m_destinationP));
		}
		operands.push_back(m_sourceA);
		operands.push_back(m_sourceB);
		operands.push_back(m_sourceC);
		operands.push_back(new HexOperand(m_memberMask));
		return operands;
	}

	// Visitors

	void Accept(ConstInstructionVisitor& visitor) const override { visitor.Visit(this); }

protected:
	DispatchMember_Type(T);

	Register<T> *m_destinationD = nullptr;
	Register<PredicateType> *m_destinationP = nullptr;
	TypedOperand<T> *m_sourceA = nullptr;
	TypedOperand<UInt32Type> *m_sourceB = nullptr;
	TypedOperand<UInt32Type> *m_sourceC = nullptr;
	uint32_t m_memberMask = 0;

	Mode m_mode;
};

DispatchImplementation(ShuffleInstruction)

}
