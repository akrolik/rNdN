#pragma once

#include "Analysis/FlowValue.h"

#include "PTX/Analysis/Framework/BackwardAnalysis.h"
#include "PTX/Traversal/ConstOperandDispatcher.h"
#include "PTX/Tree/Tree.h"

namespace PTX {
namespace Analysis {

struct LiveVariablesValue : ::Analysis::Value<std::string>
{
	using Type = std::string;
	using ::Analysis::Value<Type>::Equals;

	struct Hash
	{
		std::size_t operator()(const Type *val) const
		{
			return std::hash<Type>()(*val);
		}
	};

	static void Print(std::ostream& os, const Type *val)
	{
		os << *val;
	}
};

using LiveVariablesProperties = ::Analysis::Set<LiveVariablesValue>;

class LiveVariables : public BackwardAnalysis<LiveVariablesProperties>, public ConstOperandDispatcher<LiveVariables>
{
public:
	using Properties = LiveVariablesProperties;
	using BackwardAnalysis<LiveVariablesProperties>::BackwardAnalysis;

	// Visitors

	void Visit(const InstructionStatement *statement) override;
	void Visit(const PredicatedInstruction *instruction) override;

	// Operand dispatch

	using ConstOperandDispatcher<LiveVariables>::Visit;

	template<Bits B, class T, class S> void Visit(const DereferencedAddress<B, T, S> *address);
	template<class T> void Visit(const Register<T> *reg);

	// Flow

	Properties InitialFlow(const FunctionDefinition<VoidType> *function) const override;
	Properties TemporaryFlow(const FunctionDefinition<VoidType> *function) const override;

	Properties Merge(const Properties& s1, const Properties& s2) const override;

	std::string Name() const override { return "Live variables"; }

private:
	bool m_destination = false;
};

}
}
