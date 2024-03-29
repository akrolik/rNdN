#pragma once

#include "Analysis/FlowValue.h"

#include "PTX/Analysis/Framework/BackwardAnalysis.h"
#include "PTX/Traversal/ConstOperandVisitor.h"
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

class LiveVariables : public BackwardAnalysis<LiveVariablesProperties>, public ConstOperandVisitor
{
public:
	using Properties = LiveVariablesProperties;

	inline const static std::string Name = "Live variables";
	inline const static std::string ShortName = "live";

	LiveVariables() : BackwardAnalysis<LiveVariablesProperties>(Name, ShortName) {}

	// Visitors

	void Visit(const InstructionStatement *statement) override;
	void Visit(const PredicatedInstruction *instruction) override;

	// Operand dispatch

	bool Visit(const _DereferencedAddress *address) override;
	bool Visit(const _Register *reg) override;

	template<Bits B, class T, class S>
	void Visit(const DereferencedAddress<B, T, S> *address);

	template<class T>
	void Visit(const Register<T> *reg);

	// Flow

	Properties InitialFlow(const FunctionDefinition<VoidType> *function) const override;
	Properties TemporaryFlow(const FunctionDefinition<VoidType> *function) const override;

	Properties Merge(const Properties& s1, const Properties& s2) const override;

private:
	bool m_destination = false;
};

}
}
