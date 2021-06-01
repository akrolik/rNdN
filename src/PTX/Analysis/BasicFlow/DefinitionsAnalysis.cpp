#include "PTX/Analysis/BasicFlow/DefinitionsAnalysis.h"

namespace PTX {
namespace Analysis {

void DefinitionsAnalysis::Visit(const InstructionStatement *statement)
{
	m_currentStatement = statement;

	// Assume the first operand is the destination

	const auto operands = statement->GetOperands();
	if (operands.size() > 0)
	{
		const auto& destination = operands.at(0);
		destination->Accept(static_cast<ConstOperandDispatcher&>(*this));
	}

	m_currentStatement = nullptr;
}

template<Bits B, class T, class S>
void DefinitionsAnalysis::Visit(const DereferencedAddress<B, T, S> *address)
{
	// Dereferenced addresses are uses
}

template<class T>
void DefinitionsAnalysis::Visit(const Register<T> *reg)
{
	AddDefinition(reg->GetName(), m_currentStatement);
}

void DefinitionsAnalysis::AddDefinition(const std::string& name, const InstructionStatement *instruction)
{
	auto destination = name;
	if (auto it = this->m_analysisSet.find(&destination); it != this->m_analysisSet.end())
	{
		it->second->insert(instruction);
	}
	else
	{
		auto key = new DefinitionsAnalysisKey::Type(destination);
		auto value = new DefinitionsAnalysisValue::Type({instruction});
		this->m_analysisSet.emplace(key, value);
	}
}

}
}
