#include "PTX/Analysis/BasicFlow/LiveVariables.h"

namespace PTX {
namespace Analysis {

void LiveVariables::Visit(const InstructionStatement *statement)
{
	// Kill all destinations, add all uses
	// Assume the first operand is the destination

	m_destination = true;
	for (const auto operand : statement->GetOperands())
	{
		operand->Accept(static_cast<ConstOperandVisitor&>(*this));
		m_destination = false;
	}
}

void LiveVariables::Visit(const PredicatedInstruction *instruction)
{
	ConstVisitor::Visit(instruction);

	if (auto [predicate, negate] = instruction->GetPredicate(); predicate != nullptr)
	{
		m_destination = false;
		Visit(predicate); // predicate->Accept(*this);
	}
}

bool LiveVariables::Visit(const _DereferencedAddress *address)
{
	address->Dispatch(*this);
	return false;
}

bool LiveVariables::Visit(const _Register *reg)
{
	reg->Dispatch(*this);
	return false;
}

template<Bits B, class T, class S>
void LiveVariables::Visit(const DereferencedAddress<B, T, S> *address)
{
	auto destination = m_destination;
	m_destination = false;

	address->GetAddress()->Accept(static_cast<ConstOperandVisitor&>(*this));

	m_destination = destination;
}

template<class T>
void LiveVariables::Visit(const Register<T> *reg)
{
	const auto& name = reg->GetName();
	if (m_destination)
	{
		m_currentSet.erase(&name);
	}
	else
	{
		m_currentSet.insert(new LiveVariablesValue::Type(name));
	}
}

LiveVariables::Properties LiveVariables::InitialFlow(const FunctionDefinition<VoidType> *function) const
{
	// Initial flow is the empty set, no variables are live!

	return {};
}

LiveVariables::Properties LiveVariables::TemporaryFlow(const FunctionDefinition<VoidType> *function) const
{
	// Initial flow is the empty set, no variables are live!

	return {};
}

LiveVariables::Properties LiveVariables::Merge(const Properties& s1, const Properties& s2) const
{
	// Simple merge operation, add all non-duplicate eelements to the out set

	Properties outSet(s1);
	outSet.insert(s2.begin(), s2.end());

	return outSet;
}

}
}
