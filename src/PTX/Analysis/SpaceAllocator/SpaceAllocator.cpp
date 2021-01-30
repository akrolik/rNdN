#include "PTX/Analysis/SpaceAllocator/SpaceAllocator.h"

#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace PTX {
namespace Analysis {

void SpaceAllocator::Analyze(const FunctionDefinition<VoidType> *function)
{
	auto timeAllocation_start = Utils::Chrono::Start("Space allocation '" + function->GetName() + "'");
	m_allocation = new SpaceAllocation();
	function->Accept(*this);
	Utils::Chrono::End(timeAllocation_start);

	if (Utils::Options::IsBackend_PrintAnalysis())
	{
		Utils::Logger::LogInfo("Space Allocation: " + function->GetName());
		Utils::Logger::LogInfo(m_allocation->ToString());
	}
}

// Declarations

bool SpaceAllocator::VisitIn(const PTX::VariableDeclaration *declaration)
{
	declaration->Accept(static_cast<ConstDeclarationVisitor&>(*this));
	return false;
}

void SpaceAllocator::Visit(const PTX::_TypedVariableDeclaration *declaration)
{
	declaration->Dispatch(*this);
}

template<class T, class S>
void SpaceAllocator::Visit(const PTX::TypedVariableDeclaration<T, S> *declaration)
{
	// Add each variable declaration to the allocation

	for (const auto& name : declaration->GetNames())
	{
		for (auto i = 0u; i < name->GetCount(); ++i)
		{
			const auto string = name->GetName(i);
			if constexpr(std::is_same<S, PTX::ParameterSpace>::value)
			{
				// Add each parameter declaration to the allocation

				m_allocation->AddParameter(string, PTX::BitSize<T::TypeBits>::NumBytes);
			}
			else if constexpr(std::is_same<S, PTX::SharedSpace>::value)
			{
				// Add each shared declaration to the allocation

				if constexpr(PTX::is_array_type<T>::value)
				{
					// Array sizes, only possible for shared spaces (not parameters)

					m_allocation->AddSharedVariable(string, PTX::BitSize<T::TypeBits>::NumBytes * T::ElementCount);
				}
				else
				{
					m_allocation->AddSharedVariable(string, PTX::BitSize<T::TypeBits>::NumBytes);
				}
			}
		}
	}
}

}
}
