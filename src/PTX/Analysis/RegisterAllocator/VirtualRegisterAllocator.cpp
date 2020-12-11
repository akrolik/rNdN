#include "PTX/Analysis/RegisterAllocator/VirtualRegisterAllocator.h"

#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace PTX {
namespace Analysis {

void VirtualRegisterAllocator::Analyze(const FunctionDefinition<VoidType> *function)
{
	function->Accept(*this);

	if (Utils::Options::IsBackend_PrintAnalysis())
	{
		Utils::Logger::LogInfo("Register Allocation: " + function->GetName());
		Utils::Logger::LogInfo(m_allocation->ToString());
	}
}

// Functions

bool VirtualRegisterAllocator::VisitIn(const FunctionDefinition<VoidType> *function)
{
	m_registerOffset = 1; // R0 reserved for dummy
	m_predicateOffset = 0;
	m_allocation = new RegisterAllocation();
	return true;
}

// Declarations

bool VirtualRegisterAllocator::VisitIn(const VariableDeclaration *declaration)
{
	return declaration->DispatchIn(*this);
}

template<class T, class S>
bool VirtualRegisterAllocator::VisitIn(const TypedVariableDeclaration<T, S> *declaration)
{
	if constexpr(std::is_same<S, RegisterSpace>::value)
	{
		for (const auto& names : declaration->GetNames())
		{
			for (auto i = 0u; i < names->GetCount(); ++i)
			{
				if constexpr(std::is_same<T, PredicateType>::value)
				{
					m_allocation->AddPredicate(names->GetName(i), m_predicateOffset);
					m_predicateOffset++;
				}
				else
				{
					const auto allocations = (BitSize<T::TypeBits>::NumBytes + 3) / 4; 
					m_allocation->AddRegister(names->GetName(i), m_registerOffset, allocations);
					m_registerOffset += allocations;
				}
			}
		}
	}
	return false;
}

}
}
