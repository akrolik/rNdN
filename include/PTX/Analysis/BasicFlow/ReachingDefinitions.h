#pragma once

#include "PTX/Analysis/Framework/ForwardAnalysis.h"
#include "PTX/Traversal/ConstOperandVisitor.h"
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

struct ReachingDefinitionsValue : ::Analysis::Value<robin_hood::unordered_set<const InstructionStatement *>>
{
	using Type = robin_hood::unordered_set<const InstructionStatement *>;
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

class ReachingDefinitions : public ForwardAnalysis<ReachingDefinitionsProperties>, public ConstOperandVisitor
{
public:
	using Properties = ReachingDefinitionsProperties;

	inline const static std::string Name = "Reaching definitions";
	inline const static std::string ShortName = "rdef";

	ReachingDefinitions() : ForwardAnalysis<ReachingDefinitionsProperties>(Name, ShortName) {}

	// Visitors

	void Visit(const InstructionStatement *statement) override;

	// Operand dispatch

	bool Visit(const _Register *reg);

	template<class T>
	void Visit(const Register<T> *reg);

	// Flow

	Properties InitialFlow(const FunctionDefinition<VoidType> *function) const override;
	Properties TemporaryFlow(const FunctionDefinition<VoidType> *function) const override;

	Properties Merge(const Properties& s1, const Properties& s2) const override;

private:
	const InstructionStatement *m_currentStatement = nullptr;
};

}
}
