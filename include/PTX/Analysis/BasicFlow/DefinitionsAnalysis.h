#pragma once

#include "PTX/Analysis/Framework/FlowInsensitiveAnalysis.h"
#include "PTX/Traversal/ConstOperandVisitor.h"
#include "PTX/Utils/PrettyPrinter.h"

#include "Analysis/FlowValue.h"

namespace PTX {
namespace Analysis {

struct DefinitionsAnalysisKey : ::Analysis::Value<std::string>
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

struct DefinitionsAnalysisValue : robin_hood::unordered_set<const InstructionStatement *>
{
	using Type = robin_hood::unordered_set<const InstructionStatement *>;

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

using DefinitionsAnalysisProperties = ::Analysis::Map<DefinitionsAnalysisKey, DefinitionsAnalysisValue, false>; 

class DefinitionsAnalysis : public FlowInsensitiveAnalysis<DefinitionsAnalysisProperties>, public ConstOperandVisitor
{
public:
	using Properties = DefinitionsAnalysisProperties;

	inline const static std::string Name = "Definitions analysis";
	inline const static std::string ShortName = "def";

	DefinitionsAnalysis() : FlowInsensitiveAnalysis<DefinitionsAnalysisProperties>(Name, ShortName) {}

	// Visitors

	void Visit(const InstructionStatement *statement) override;

	// Operand dispatch

	bool Visit(const _DereferencedAddress *address);
	bool Visit(const _Register *reg);

	template<class T>
	void Visit(const Register<T> *reg);

	// Definitions

	void AddDefinition(const std::string& name, const InstructionStatement *instruction);

private:
	const InstructionStatement *m_currentStatement = nullptr;
};

}
}
