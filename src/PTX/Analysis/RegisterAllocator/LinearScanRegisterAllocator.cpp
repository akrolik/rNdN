#include "PTX/Analysis/RegisterAllocator/LinearScanRegisterAllocator.h"

#include <array>

#include "SASS/Tree/Tree.h"

#include "Utils/Chrono.h"

namespace PTX {
namespace Analysis {

void LinearScanRegisterAllocator::Analyze(const FunctionDefinition<VoidType> *function)
{
	auto& functionName = function->GetName();

	auto timeAllocation_start = Utils::Chrono::Start(Name + " '" + functionName + "'");

	// Collect all register sizes

	function->Accept(*this);

	// Initialize blank register allocation, depending on the architecture

	if (SASS::Volta::IsSupported(m_computeCapability))
	{
		m_allocation = new RegisterAllocation();
	}
	else
	{
		m_allocation = new RegisterAllocation(RegisterAllocation::MaxRegisters - 2);
	}

	// Construct vector of live interval tuples (name, start, end)

	const auto& inputIntervals = m_liveIntervals.GetLiveIntervals();

	std::vector<std::tuple<std::string, unsigned int, unsigned int>> liveIntervals;
	liveIntervals.reserve(inputIntervals.size());

	for (const auto& [name, interval] : inputIntervals)
	{
		liveIntervals.emplace_back(name, interval.first, interval.second);
	}
	
	// Sort live intervals by start position

	std::sort(liveIntervals.begin(), liveIntervals.end(), [](auto &left, auto &right)
	{
		auto& leftValue = std::get<1>(left);
		auto& rightValue = std::get<1>(right);
		if (leftValue != rightValue)
		{
			return leftValue < rightValue;
		}

		return std::get<0>(left) < std::get<0>(right);
	});

	// Initialize empty vector of active records (end, register, range). Range=0 for predicates

	std::vector<std::tuple<unsigned int, std::uint8_t, std::uint8_t>> activeIntervals;
	activeIntervals.reserve(liveIntervals.size());

	// Keep track of allocated registers and predicates

	std::array<bool, RegisterAllocation::MaxPredicates> predicateMap{};
	std::array<bool, RegisterAllocation::MaxRegisters> registerMap{};

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

			auto maxPredicates = m_allocation->GetMaxPredicates();
			while (intervalRegister < maxPredicates)
			{
				if (predicateMap[intervalRegister] == false)
				{
					break;
				}
				intervalRegister++;
			}

			// Check free predicate register found

			if (intervalRegister >= maxPredicates)
			{
				Utils::Logger::LogError("Linear scan exceeded max predicate count (" + std::to_string(maxPredicates) + ") for function '" + function->GetName() + "'");
			}

			// Allocate predicate register

			predicateMap[intervalRegister] = true;
			m_allocation->AddPredicate(intervalName, intervalRegister);
		}
		else
		{
			intervalRange = (DynamicBitSize::GetByte(intervalBits) + 3) / 4;

			auto maxRegisters = m_allocation->GetMaxRegisters();
			while (intervalRegister < maxRegisters)
			{
				// Check to make sure register range fits at this position
				// -1 to account for current position in range

				if (intervalRegister + intervalRange - 1 >= maxRegisters)
				{
					intervalRegister = maxRegisters;
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

			if (intervalRegister >= maxRegisters)
			{
				Utils::Logger::LogError("Linear scan exceeded max register count (" + std::to_string(maxRegisters) + ") for function '" + function->GetName() + "'");
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
	Utils::Chrono::End(timeAllocation_start);

	if (Utils::Options::IsBackend_PrintAnalysis(ShortName, functionName))
	{
		Utils::Logger::LogInfo(Name + " '" + functionName + "'");
		Utils::Logger::LogInfo(m_allocation->ToString());
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
