#include "PTX/Analysis/SpaceAllocator/LocalSpaceAllocator.h"

#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace PTX {
namespace Analysis {

void LocalSpaceAllocator::Analyze(const FunctionDefinition<VoidType> *function)
{
	auto timeAllocation_start = Utils::Chrono::Start("Local space allocation '" + function->GetName() + "'");
	m_allocation = new LocalSpaceAllocation(m_globalAllocation->GetSharedMemorySize());
	function->Accept(*this);
	Utils::Chrono::End(timeAllocation_start);

	if (Utils::Options::IsBackend_PrintAnalysis())
	{
		Utils::Logger::LogInfo("Local Space Allocation: " + function->GetName());
		Utils::Logger::LogInfo(m_allocation->ToString());
	}
}

// Declarations

bool LocalSpaceAllocator::VisitIn(const VariableDeclaration *declaration)
{
	declaration->Accept(static_cast<ConstDeclarationVisitor&>(*this));
	return false;
}

void LocalSpaceAllocator::Visit(const _TypedVariableDeclaration *declaration)
{
	declaration->Dispatch(*this);
}

template<class T, class S>
void LocalSpaceAllocator::Visit(const TypedVariableDeclaration<T, S> *declaration)
{
	// Add each variable declaration to the allocation

	for (const auto& name : declaration->GetNames())
	{
		for (auto i = 0u; i < name->GetCount(); ++i)
		{
			const auto string = name->GetName(i);
			if constexpr(std::is_same<S, ParameterSpace>::value)
			{
				// Add each parameter declaration to the allocation

				m_allocation->AddParameter(string, BitSize<T::TypeBits>::NumBytes);
			}
			else if constexpr(std::is_same<S, SharedSpace>::value)
			{
				// Add each shared declaration to the allocation

				if constexpr(is_array_type<T>::value)
				{
					// Array sizes, only possible for shared spaces (not parameters)

					m_allocation->AddSharedMemory(string, BitSize<T::TypeBits>::NumBytes * T::ElementCount);
				}
				else
				{
					m_allocation->AddSharedMemory(string, BitSize<T::TypeBits>::NumBytes);
				}
			}
		}
	}
}

}
}
