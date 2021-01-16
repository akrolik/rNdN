#include "PTX/Analysis/BasicFlow/LiveVariables.h"

namespace PTX {
namespace Analysis {

void LiveVariables::Visit(const InstructionStatement *statement)
{
	// Kill all destinations, add all uses

	m_currentInSet = m_currentOutSet;

	m_destination = true;
	for (const auto operand : statement->GetOperands())
	{
		operand->Accept(static_cast<ConstOperandDispatcher&>(*this));
		m_destination = false;
	}

	//TODO: Predicated instructions
}

template<Bits B, class T, class S>
void LiveVariables::Visit(const DereferencedAddress<B, T, S> *address)
{
	auto destination = m_destination;
	m_destination = false;

	address->GetAddress()->Accept(static_cast<ConstOperandDispatcher&>(*this));

	m_destination = destination;
}

template<Bits B, class T, class S>
void LiveVariables::Visit(const RegisterAddress<B, T, S> *address)
{
	address->GetRegister()->Accept(static_cast<ConstOperandDispatcher&>(*this));
}

template<class T>
void LiveVariables::Visit(const Register<T> *reg)
{
	const auto& name = reg->GetName();
	if (m_destination)
	{
		m_currentInSet.erase(&name);
	}
	else
	{
		m_currentInSet.insert(new LiveVariablesValue::Type(name));
	}
}

LiveVariables::Properties LiveVariables::InitialFlow() const
{
	// Initial flow is the empty set, no variables are live!

	return {};
}

LiveVariables::Properties LiveVariables::TemporaryFlow() const
{
	// Initial flow is the empty set, no variables are live!

	return {};
}

LiveVariables::Properties LiveVariables::Merge(const Properties& s1, const Properties& s2) const
{
	// Simple merge operation, add all non-duplicate eelements to the out set

	Properties outSet;

	outSet.insert(s1.begin(), s1.end());
	outSet.insert(s2.begin(), s2.end());

	return outSet;
}

}
}