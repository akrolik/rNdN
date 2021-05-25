#include "PTX/Analysis/RegisterAllocator/LinearScanRegisterAllocator.h"

#include <array>

#include "Utils/Chrono.h"

namespace PTX {
namespace Analysis {

void LinearScanRegisterAllocator::Analyze(const FunctionDefinition<VoidType> *function)
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

void LinearScanRegisterAllocator::VisitOut(const FunctionDefinition<VoidType>* function)
{
	// Initialize blank register allocation

	m_allocation = new RegisterAllocation();

	// Construct vector of live interval tuples (name, start, end)

	std::vector<std::tuple<std::string, unsigned int, unsigned int>> liveIntervals;
	for (const auto& [name, interval] : m_liveIntervals.GetLiveIntervals())
	{
		liveIntervals.emplace_back(name, interval.first, interval.second);
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

	// For each live interval, perform allocation

	for (const auto& [intervalName, intervalStart, intervalEnd] : liveIntervals)
	{
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
				for (auto i = activeRegister; i < activeRegister + activeRange; ++i)
				{
					registerMap[i] = false;
				}
			}

			// Remove expired interval (must occur after the above)
			
			it = activeIntervals.erase(it);
		}

		// Assign register to variable, taking sizes into account

		auto intervalRegister = 0u;
		auto intervalRange = 1u;

		const auto intervalBits = m_registerBits.at(intervalName);
		if (intervalBits == Bits::Bits1)
		{
			intervalRange = 0u; // Predicates have no range

			// Find open predicate register

			while (intervalRegister < RegisterAllocation::MaxPredicate)
			{
				if (predicateMap[intervalRegister] == false)
				{
					break;
				}
				intervalRegister++;
			}

			// Check free predicate register found

			if (intervalRegister >= RegisterAllocation::MaxPredicate)
			{
				Utils::Logger::LogError("Linear scan exceeded max predicate count (" + std::to_string(RegisterAllocation::MaxPredicate) + ") for function '" + function->GetName() + "'");
			}

			// Allocate predicate register

			predicateMap[intervalRegister] = true;
			m_allocation->AddPredicate(intervalName, intervalRegister);
		}
		else
		{
			intervalRange = (DynamicBitSize::GetByte(intervalBits) + 3) / 4;

			while (intervalRegister < RegisterAllocation::MaxRegister)
			{
				// Check to make sure register range fits at this position
				// -1 to account for current position in range

				if (intervalRegister + intervalRange - 1 >= RegisterAllocation::MaxRegister)
				{
					intervalRegister = RegisterAllocation::MaxRegister;
					break;
				}

				// Find if range is free

				auto valid = true;
				for (auto j = intervalRegister; j < intervalRegister + intervalRange; ++j)
				{
					if (registerMap[j] == true)
					{
						valid = false;
					}
				}

				if (valid)
				{
					break;
				}
				intervalRegister += intervalRange;
			}

			// Check free register found

			if (intervalRegister >= RegisterAllocation::MaxRegister)
			{
				Utils::Logger::LogError("Linear scan exceeded max register count (" + std::to_string(RegisterAllocation::MaxRegister) + ") for function '" + function->GetName() + "'");
			}

			// Allocate register for range

			for (auto i = intervalRegister; i < intervalRegister + intervalRange; ++i)
			{
				registerMap[i] = true;
			}
			m_allocation->AddRegister(intervalName, intervalRegister, intervalRange);
		}

		// Add interval to active set, sorting by increasing end point

		auto inserted = false;
		for (auto it = std::begin(activeIntervals); it != std::end(activeIntervals); ++it)
		{
			const auto activeEnd = std::get<0>(*it);
			if (intervalEnd < activeEnd)
			{
				inserted = true;
				activeIntervals.emplace(it, intervalEnd, intervalRegister, intervalRange);
				break;
			}
		}

		if (inserted == false)
		{
			activeIntervals.emplace_back(intervalEnd, intervalRegister, intervalRange);
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
				m_registerBits.emplace(names->GetName(i), T::TypeBits);
			}
		}
	}
}

}
}
