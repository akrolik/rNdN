#pragma once

#include "PTX/Tree/Instructions/PredicatedInstruction.h"
#include "PTX/Tree/Instructions/Modifiers/ScopeModifier.h"

namespace PTX {

class FenceInstruction : public PredicatedInstruction, public ScopeModifier<>
{
public:
	using Scope = ScopeModifier<>::Scope;

	enum class Synchronization {
		None,
		AcquireRelease,
		SequentialConsistency
	};

	static std::string SynchronizationString(Synchronization synchronization)
	{
		switch (synchronization)
		{
			case Synchronization::None:
				return "";
			case Synchronization::AcquireRelease:
				return ".acq_rel";
			case Synchronization::SequentialConsistency:
				return ".sc";
		}
		return ".<unknown>";
	}

	FenceInstruction(Scope scope, Synchronization synchronization = Synchronization::None) : ScopeModifier<>(scope), m_synchronization(synchronization) {}

	Synchronization GetSynchronization() const { return m_synchronization; }
	void SetSynchronization(Synchronization synchronization) { m_synchronization = synchronization; }

	static std::string Mnemonic() { return "fence"; }

	std::string OpCode() const override
	{
		return Mnemonic() + SynchronizationString(m_synchronization) + ScopeModifier<>::OpCodeModifier();
	}

protected:
	Synchronization m_synchronization = Synchronization::None;
};

}
