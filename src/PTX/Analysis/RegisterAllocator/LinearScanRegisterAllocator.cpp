#include "PTX/Analysis/RegisterAllocator/LinearScanRegisterAllocator.h"

#include <array>

namespace PTX {
namespace Analysis {

void LinearScanRegisterAllocator::Analyze(const FunctionDefinition<VoidType> *function)
{
	auto timeAllocation_start = Utils::Chrono::Start("Register allocation '" + function->GetName() + "'");
	function->Accept(*this);
	Utils::Chrono::End(timeAllocation_start);

	if (Utils::Options::IsBackend_PrintAnalysis())
	{
		Utils::Logger::LogInfo("Register Allocation: " + function->GetName());
		Utils::Logger::LogInfo(m_allocation->ToString());
	}
}

void LinearScanRegisterAllocator::VisitOut(const FunctionDefinition<VoidType>* function)
{
	// Initialize blank register allocation

	m_allocation = new RegisterAllocation();

	// Construct vector of live interval tuples (name, start, end)

	std::vector<std::tuple<std::string, unsigned int, unsigned int>> liveIntervals;
	for (const auto& [name, interval] : m_liveIntervals.GetLiveIntervals())
	{
		liveIntervals.push_back({ name, interval.first, interval.second });
	}
	
	// Sort live intervals by start position

	std::sort(liveIntervals.begin(), liveIntervals.end(), [](auto &left, auto &right)
	{
		return std::get<1>(left) < std::get<1>(right);
	});

	// Initialize empty vector of active records (end, register, range). Range=0 for predicates

	std::vector<std::tuple<unsigned int, std::uint8_t, std::uint8_t>> activeIntervals;

	// Keep track of allocated registers and predicates

	std::array<bool, RegisterAllocation::MaxPredicate> predicateMap{};
	std::array<bool, RegisterAllocation::MaxRegister> registerMap{};

	auto temporaryCount = 6u; // Dummy R0-R5
	for (auto i = 0u; i < temporaryCount; ++i)
	{
		registerMap[i] = true;
	}
	m_allocation->SetTemporaryRegisters(0, temporaryCount);

	// For each live interval, perform allocation

	for (const auto& [intervalName, intervalStart, intervalEnd] : liveIntervals)
	{
		//TODO: Update debug code
		// std::cout << intervalName << std::endl;
		// std::cout << "  - Input register map:";
		// for (auto i = 0; i < 20; ++i)
		// {
		// 	std::cout << " " << registerMap[i];
		// }
		// std::cout << std::endl;

		// std::cout << "  - Input active set:";
		// for (const auto& [end, reg, range] : activeIntervals)
		// {
		// 	std::cout << " (" << end << ", " << (unsigned int)reg << ", " << (unsigned int)range << ")";
		// }
		// std::cout << std::endl;

		// Free expired intervals

		for (auto it = std::begin(activeIntervals); it != std::end(activeIntervals); /* do nothing */)
		{
			// Check if active interval is still live. If so, since the active intervals
			// are sorted by end position, we can progress directly to allocation

			const auto activeEnd = std::get<0>(*it);
			if (activeEnd >= intervalStart)
			{
				break;
			}

			// Remove the allocation from the register or predicate map (range is 0 for predicates)

			const auto activeRegister = std::get<1>(*it);
			const auto activeRange = std::get<2>(*it);

			if (activeRange == 0)
			{
				predicateMap[activeRegister] = false;
			}
			else
			{
				// std::cout << "  - Free register range: " << (unsigned int)activeRegister << ", " << (unsigned int)activeRange << std::endl;
				for (auto i = activeRegister; i < activeRegister + activeRange; ++i)
				{
					registerMap[i] = false;
				}
			}

			// Remove expired interval (must occur after the above)
			
			activeIntervals.erase(it);
		}

		// std::cout << "  - Clean register map:";
		// for (auto i = 0; i < 20; ++i)
		// {
		// 	std::cout << " " << registerMap[i];
		// }
		// std::cout << std::endl;

		// std::cout << "  - Clean active set:";
		// for (const auto& [end, reg, range] : activeIntervals)
		// {
		// 	std::cout << " (" << end << ", " << (unsigned int)reg << ", " << (unsigned int)range << ")";
		// }
		// std::cout << std::endl;

		// Assign register to variable, taking sizes into account

		auto intervalRegister = 0u;
		auto intervalRange = 1u;

		const auto intervalBits = m_registerBits.at(intervalName);
		if (intervalBits == Bits::Bits1)
		{
			// Find open predicate register

			for (auto i = 0; i < RegisterAllocation::MaxPredicate; ++i)
			{
				if (predicateMap[i] == false)
				{
					intervalRegister = i;
					intervalRange = 0u; // Predicates have no range
					break;
				}
			}

			// Check free predicate register found

			if (intervalRegister == RegisterAllocation::MaxPredicate)
			{
				Utils::Logger::LogError("Linear scan exceeded predicate count (" + std::to_string(RegisterAllocation::MaxPredicate) + ") for function '" + function->GetName() + "'");
			}

			// Allocate predicate register

			predicateMap[intervalRegister] = true;
			m_allocation->AddPredicate(intervalName, intervalRegister);
		}
		else
		{
			intervalRange = (DynamicBitSize::GetByte(intervalBits) + 3) / 4;

			for (auto i = 0; i < RegisterAllocation::MaxRegister; i += intervalRange)
			{
				// Check to make sure register range fits at this position
				// -1 to account for current position in range

				if (i + intervalRange - 1 >= RegisterAllocation::MaxRegister)
				{
					break;
				}

				// Find if range is free

				auto valid = true;
				for (auto j = i; j < i + intervalRange; ++j)
				{
					if (registerMap[j] == true)
					{
						valid = false;
					}
				}

				if (valid)
				{
					intervalRegister = i;
					break;
				}
			}

			// Check free register found

			if (intervalRegister == RegisterAllocation::MaxRegister)
			{
				Utils::Logger::LogError("Linear scan exceeded register count (" + std::to_string(RegisterAllocation::MaxRegister) + ") for function '" + function->GetName() + "'");
			}

			// Allocate register for range

			for (auto i = intervalRegister; i < intervalRegister + intervalRange; ++i)
			{
				registerMap[i] = true;
			}
			m_allocation->AddRegister(intervalName, intervalRegister, intervalRange);

			// std::cout << "  - Allocate register range: " << (unsigned int)intervalRegister << ", " << (unsigned int)intervalRange << std::endl;
		}

		// std::cout << "  - Allocated register map:";
		// for (auto i = 0; i < 20; ++i)
		// {
		// 	std::cout << " " << registerMap[i];
		// }
		// std::cout << std::endl;

		// Add interval to active set, sorting by increasing end point

		auto inserted = false;
		for (auto it = std::begin(activeIntervals); it != std::end(activeIntervals); ++it)
		{
			const auto activeEnd = std::get<0>(*it);
			if (intervalEnd < activeEnd)
			{
				inserted = true;
				activeIntervals.insert(it, { intervalEnd, intervalRegister, intervalRange });
				break;
			}
		}

		if (inserted == false)
		{
			activeIntervals.push_back({ intervalEnd, intervalRegister, intervalRange });
		}
	}
}

bool LinearScanRegisterAllocator::VisitIn(const VariableDeclaration *declaration)
{
	declaration->Accept(static_cast<ConstDeclarationVisitor&>(*this));
	return false;
}

void LinearScanRegisterAllocator::Visit(const _TypedVariableDeclaration *declaration)
{
	declaration->Dispatch(*this);
}

template<class T, class S>
void LinearScanRegisterAllocator::Visit(const TypedVariableDeclaration<T, S> *declaration)
{
	if constexpr(std::is_same<S, RegisterSpace>::value)
	{
		for (const auto& names : declaration->GetNames())
		{
			for (auto i = 0u; i < names->GetCount(); ++i)
			{
				m_registerBits[names->GetName(i)] = T::TypeBits;
			}
		}
	}
}

}
}
