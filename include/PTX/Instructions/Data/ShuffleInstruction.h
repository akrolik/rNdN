#pragma once

#include "PTX/Instructions/PredicatedInstruction.h"

#include "PTX/Operands/Operand.h"
#include "PTX/Operands/Extended/DualOperand.h"
#include "PTX/Operands/Extended/HexOperand.h"
#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class T, bool Assert = true>
class ShuffleInstruction : public PredicatedInstruction
{
public:
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

	ShuffleInstruction(const Register<T> *destinationD, const TypedOperand<T> *sourceA, const TypedOperand<UInt32Type> *sourceB, const TypedOperand<UInt32Type> *sourceC, uint32_t memberMask, Mode mode) : m_destinationD(destinationD), m_sourceA(sourceA), m_sourceB(sourceB), m_sourceC(sourceC), m_memberMask(memberMask), m_mode(mode) {}

	const Register<T> *GetDestination() const { return m_destinationD; }
	void SetDestination(const Register<T> *destination) { m_destinationD = destination; }

	const Register<PredicateType> *GetDestinationP() const { return m_destinationP; }
	void SetDestinationP(const Register<PredicateType> *destination) { m_destinationP = destination; }

	const TypedOperand<T> *GetSourceA() const { return m_sourceA; }
	void SetSourceA(const TypedOperand<T> *source) { m_sourceA = source; }

	Mode GetMode() const { return m_mode; }
	void SetMode(Mode mode) { m_mode = mode; }

	const TypedOperand<UInt32Type> *GetSourceB() const { return m_sourceB; }
	void SetSourceB(const TypedOperand<UInt32Type> *source) { m_sourceB = source; }

	const TypedOperand<UInt32Type> *GetSourceC() const { return m_sourceC; }
	void SetSourceC(const TypedOperand<UInt32Type> *source) { m_sourceC = source; }

	uint32_t GetMemberMask() const { return m_memberMask; }
	void SetMemberMask(uint32_t memberMask) { m_memberMask = memberMask; }

	static std::string Mnemonic() { return "shfl"; }

	std::string OpCode() const override
	{
		return Mnemonic() + ".sync" + ModeString(m_mode) + Bit32Type::Name();
	}

	std::vector<const Operand *> Operands() const override
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

private:
	const Register<T> *m_destinationD = nullptr;
	const Register<PredicateType> *m_destinationP = nullptr;
	const TypedOperand<T> *m_sourceA = nullptr;
	const TypedOperand<UInt32Type> *m_sourceB = nullptr;
	const TypedOperand<UInt32Type> *m_sourceC = nullptr;
	uint32_t m_memberMask = 0;

	Mode m_mode;
};

}
