#include "PTX/Analysis/SpaceAllocator/ParameterSpaceAllocator.h"

#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace PTX {
namespace Analysis {

void ParameterSpaceAllocator::Analyze(const FunctionDefinition<VoidType> *function)
{
	auto timeAllocation_start = Utils::Chrono::Start("Function space allocation '" + function->GetName() + "'");
	m_allocation = new ParameterSpaceAllocation();
	function->Accept(*this);
	Utils::Chrono::End(timeAllocation_start);

	if (Utils::Options::IsBackend_PrintAnalysis())
	{
		Utils::Logger::LogInfo("Function Space Allocation: " + function->GetName());
		Utils::Logger::LogInfo(m_allocation->ToString());
	}
}

// Declarations

bool ParameterSpaceAllocator::VisitIn(const FunctionDefinition<VoidType> *function)
{
	// Traverse only parameters

	for (const auto& parameter : function->GetParameters())
	{
		parameter->Accept(static_cast<ConstHierarchicalVisitor&>(*this));
	}
	return false;
}

bool ParameterSpaceAllocator::VisitIn(const VariableDeclaration *declaration)
{
	declaration->Accept(static_cast<ConstDeclarationVisitor&>(*this));
	return false;
}

void ParameterSpaceAllocator::Visit(const _TypedVariableDeclaration *declaration)
{
	declaration->Dispatch(*this);
}

template<class T, class S>
void ParameterSpaceAllocator::Visit(const TypedVariableDeclaration<T, S> *declaration)
{
	// Add each parameter declaration to the allocation

	for (const auto& name : declaration->GetNames())
	{
		for (auto i = 0u; i < name->GetCount(); ++i)
		{
			if constexpr(std::is_same<S, ParameterSpace>::value)
			{
				const auto string = name->GetName(i);
				const auto dataSize = BitSize<T::TypeBits>::NumBytes;

				m_allocation->AddParameter(string, dataSize);
			}
		}
	}
}

}
}