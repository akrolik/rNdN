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

	ShuffleInstruction(const Register<T> *destinationD, const TypedOperand<T> *sourceA, Mode mode, const TypedOperand<Bit32Type> *sourceB, const TypedOperand<Bit32Type> *sourceC, unsigned int memberMask) : m_destinationD(destinationD), m_sourceA(sourceA), m_mode(mode), m_sourceB(sourceB), m_sourceC(sourceC), m_memberMask(memberMask) {}

	std::string OpCode() const override
	{
		return "shfl.sync" + ModeString(m_mode) + ".b32";
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
	//TODO: Verify types of operands, as well as the membermask constant vs register
	const Register<T> *m_destinationD = nullptr;
	const Register<PredicateType> *m_destinationP = nullptr;

	Mode m_mode;

	const TypedOperand<T> *m_sourceA = nullptr;
	const TypedOperand<Bit32Type> *m_sourceB = nullptr;
	const TypedOperand<Bit32Type> *m_sourceC = nullptr;
	unsigned int m_memberMask = 0;
};

}
