#include "PTX/Analysis/RegisterAllocator/VirtualRegisterAllocator.h"

#include "Utils/Logger.h"

namespace PTX {
namespace Analysis {

void VirtualRegisterAllocator::Analyze(const Program *program)
{
	program->Accept(*this);
}

// Functions

bool VirtualRegisterAllocator::VisitIn(const FunctionDefinition<VoidType> *function)
{
	m_registerOffset = 1; // R0 reserved for dummy
	m_predicateOffset = 0;
	m_currentAllocation = new RegisterAllocation();
	return true;
}

void VirtualRegisterAllocator::VisitOut(const FunctionDefinition<VoidType> *function)
{
	Utils::Logger::LogInfo("Register Allocation: " + function->GetName());
	Utils::Logger::LogInfo(m_currentAllocation->ToString());

	m_allocations[function] = m_currentAllocation;
	m_currentAllocation = nullptr;
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
					m_currentAllocation->AddPredicate(names->GetName(i), m_predicateOffset);
					m_predicateOffset++;
				}
				else
				{
					m_currentAllocation->AddRegister(names->GetName(i), m_registerOffset);

					const auto allocations = (BitSize<T::TypeBits>::NumBytes + 3) / 4; 
					m_registerOffset += allocations;
				}
			}
		}
	}
	return false;
}

}
}
