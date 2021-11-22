#include "PTX/Analysis/SpaceAllocator/ParameterSpaceAllocator.h"

#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace PTX {
namespace Analysis {

void ParameterSpaceAllocator::Analyze(const FunctionDefinition<VoidType> *function)
{
	auto& functionName = function->GetName();

	auto timeAllocation_start = Utils::Chrono::Start(Name + " '" + functionName + "'");

	auto parameterOffset = GetParameterOffset();
	m_allocation = new ParameterSpaceAllocation(parameterOffset);

	for (const auto& parameter : function->GetParameters())
	{
		parameter->Accept(static_cast<ConstHierarchicalVisitor&>(*this));
	}

	Utils::Chrono::End(timeAllocation_start);

	if (Utils::Options::IsBackend_PrintAnalysis(ShortName, functionName))
	{
		Utils::Logger::LogInfo(Name + " '" + functionName + "'");
		Utils::Logger::LogInfo(m_allocation->ToString());
	}
}

// Declarations

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

std::size_t ParameterSpaceAllocator::GetParameterOffset() const
{
	if (SASS::Maxwell::IsSupported(m_computeCapability))
	{
		return SASS::Maxwell::CBANK_PARAM_OFFSET;
	}
	else if (SASS::Volta::IsSupported(m_computeCapability))
	{
		return SASS::Volta::CBANK_PARAM_OFFSET;
	}
	Utils::Logger::LogError("Unsupported CUDA compute capability for parameter allocation 'sm_" + std::to_string(m_computeCapability) + "'");
}

}
}
