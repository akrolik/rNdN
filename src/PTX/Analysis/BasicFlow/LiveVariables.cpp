#include "PTX/Analysis/BasicFlow/LiveVariables.h"

namespace PTX {
namespace Analysis {

void LiveVariables::Visit(const InstructionStatement *statement)
{
	// Kill all destinations, add all uses

	m_currentInSet = m_currentOutSet;

	// Assume the first operand is the destination

	m_destination = true;
	for (const auto operand : statement->GetOperands())
	{
		operand->Accept(static_cast<ConstOperandDispatcher&>(*this));
		m_destination = false;
	}
}

void LiveVariables::Visit(const PredicatedInstruction *instruction)
{
	ConstVisitor::Visit(instruction);

	if (auto [predicate, negate] = instruction->GetPredicate(); predicate != nullptr)
	{
		m_destination = false;
		predicate->Accept(static_cast<ConstOperandDispatcher&>(*this));
	}
}

template<Bits B, class T, class S>
void LiveVariables::Visit(const DereferencedAddress<B, T, S> *address)
{
	auto destination = m_destination;
	m_destination = false;

	address->GetAddress()->Accept(static_cast<ConstOperandDispatcher&>(*this));

	m_destination = destination;
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

//TODO: Indexed registers

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
