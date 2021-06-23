#include "PTX/Analysis/RegisterAllocator/VirtualRegisterAllocator.h"

#include "Utils/Chrono.h"
#include "Utils/Logger.h"
#include "Utils/Options.h"

namespace PTX {
namespace Analysis {

void VirtualRegisterAllocator::Analyze(const FunctionDefinition<VoidType> *function)
{
	auto& functionName = function->GetName();

	auto timeAllocation_start = Utils::Chrono::Start(Name + " '" + functionName + "'");
	function->Accept(*this);
	Utils::Chrono::End(timeAllocation_start);

	if (Utils::Options::IsBackend_PrintAnalysis(ShortName, functionName))
	{
		Utils::Logger::LogInfo(Name + " '" + functionName + "'");
		Utils::Logger::LogInfo(m_allocation->ToString());
	}
}

// Functions

bool VirtualRegisterAllocator::VisitIn(const FunctionDefinition<VoidType> *function)
{
	// Register allocation

	m_currentFunction = function;
	m_registerOffset = 1; // R0 reserved for dummy
	m_predicateOffset = 0;
	m_allocation = new RegisterAllocation();

	return true;
}

void VirtualRegisterAllocator::VisitOut(const FunctionDefinition<VoidType> *function)
{
	m_currentFunction = nullptr;
}

// Declarations

bool VirtualRegisterAllocator::VisitIn(const VariableDeclaration *declaration)
{
	declaration->Accept(static_cast<ConstDeclarationVisitor&>(*this));
	return false;
}

void VirtualRegisterAllocator::Visit(const _TypedVariableDeclaration *declaration)
{
	declaration->Dispatch(*this);
}

template<class T, class S>
void VirtualRegisterAllocator::Visit(const TypedVariableDeclaration<T, S> *declaration)
{
	if constexpr(std::is_same<S, RegisterSpace>::value)
	{
		for (const auto& names : declaration->GetNames())
		{
			for (auto i = 0u; i < names->GetCount(); ++i)
			{
				if constexpr(std::is_same<T, PredicateType>::value)
				{
					if (auto maxPredicates = m_allocation->GetMaxPredicates(); m_predicateOffset >= maxPredicates)
					{
						Utils::Logger::LogError("Virtual allocation exceeded max predicate count (" + std::to_string(maxPredicates) + ") for function '" + m_currentFunction->GetName() + "'");
					}

					m_allocation->AddPredicate(names->GetName(i), m_predicateOffset);
					m_predicateOffset++;
				}
				else
				{
					if (auto maxRegisters = m_allocation->GetMaxRegisters(); m_registerOffset >= maxRegisters)
					{
						Utils::Logger::LogError("Virtual allocation exceeded max register count (" + std::to_string(maxRegisters) + ") for function '" + m_currentFunction->GetName() + "'");
					}

					const auto allocations = (BitSize<T::TypeBits>::NumBytes + 3) / 4; 
					m_allocation->AddRegister(names->GetName(i), m_registerOffset, allocations);
					m_registerOffset += allocations;
				}
			}
		}
	}
}

}
}
