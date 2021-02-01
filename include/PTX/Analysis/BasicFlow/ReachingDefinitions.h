#pragma once

#include "PTX/Analysis/Framework/ForwardAnalysis.h"
#include "PTX/Traversal/ConstOperandDispatcher.h"
#include "PTX/Utils/PrettyPrinter.h"

#include "Analysis/FlowValue.h"

namespace PTX {
namespace Analysis {

struct ReachingDefinitionsKey : ::Analysis::Value<std::string>
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

struct ReachingDefinitionsValue : ::Analysis::Value<std::unordered_set<const InstructionStatement *>>
{
	using Type = std::unordered_set<const InstructionStatement *>;
	using ::Analysis::Value<Type>::Equals;

	static void Print(std::ostream& os, const Type *val)
	{
		os << "[";
		bool first = true;
		for (const auto& statement : *val)
		{
			if (!first)
			{
				os << ", ";
			}
			first = false;
			os << PrettyPrinter::PrettyString(statement, true);
		}
		os << "]";
	}
};

using ReachingDefinitionsProperties = ::Analysis::Map<ReachingDefinitionsKey, ReachingDefinitionsValue>; 

class ReachingDefinitions : public ForwardAnalysis<ReachingDefinitionsProperties>, public ConstOperandDispatcher<ReachingDefinitions>
{
public:
	using Properties = ReachingDefinitionsProperties;
	using ForwardAnalysis<ReachingDefinitionsProperties>::ForwardAnalysis;

	void Visit(const InstructionStatement *statement) override;

	// Operand dispatch

	using ConstOperandDispatcher<ReachingDefinitions>::Visit;

	template<class T> void Visit(const Register<T> *reg);

	// Flow

	Properties InitialFlow() const override;
	Properties TemporaryFlow() const override;

	Properties Merge(const Properties& s1, const Properties& s2) const override;

	std::string Name() const override { return "Reaching definitions"; }

private:
	const InstructionStatement *m_currentStatement = nullptr;
};

}
}
