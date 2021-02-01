#include "PTX/Analysis/SpaceAllocator/GlobalSpaceAllocator.h"

#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace PTX {
namespace Analysis {

void GlobalSpaceAllocator::Analyze(const Module *module)
{
	auto timeAllocation_start = Utils::Chrono::Start("Global space allocation");
	m_allocation = new GlobalSpaceAllocation();
	module->Accept(*this);
	Utils::Chrono::End(timeAllocation_start);

	if (Utils::Options::IsBackend_PrintAnalysis())
	{
		Utils::Logger::LogInfo("Global Space Allocation");
		Utils::Logger::LogInfo(m_allocation->ToString());
	}
}

// Declarations

bool GlobalSpaceAllocator::VisitIn(const VariableDeclaration *declaration)
{
	declaration->Accept(static_cast<ConstDeclarationVisitor&>(*this));
	return false;
}

void GlobalSpaceAllocator::Visit(const _TypedVariableDeclaration *declaration)
{
	declaration->Dispatch(*this);
}

template<class T, class S>
void GlobalSpaceAllocator::Visit(const TypedVariableDeclaration<T, S> *declaration)
{
	// Add each variable declaration to the allocation

	for (const auto& name : declaration->GetNames())
	{
		for (auto i = 0u; i < name->GetCount(); ++i)
		{
			const auto string = name->GetName(i);
			if constexpr(std::is_same<S, GlobalSpace>::value)
			{
				//TODO: Global variables
			}
			else if constexpr(std::is_same<S, SharedSpace>::value)
			{
				if (declaration->GetLinkDirective() == Declaration::LinkDirective::External)
				{
					// External shared memory is dynamically defined

					m_allocation->AddDynamicSharedMemory(string);
				}
				else
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

// Functions

bool GlobalSpaceAllocator::VisitIn(const FunctionDefinition<VoidType> *function)
{
	// End the traversal, only handle module defined variables

	return false;
}

}
}
