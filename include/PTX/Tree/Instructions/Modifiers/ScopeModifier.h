#pragma once

#include <iostream>

namespace PTX {

template<bool Required = true>
class ScopeModifier
{
public:
	enum class Scope {
		None,
		CTA,
		GPU,
		System
	};

	static std::string ScopeString(Scope scope)
	{
		switch (scope)
		{
			case Scope::None:
				return "";
			case Scope::CTA:
				return ".cta";
			case Scope::GPU:
				return ".gpu";
			case Scope::System:
				return ".sys";
		}
		return ".<unknown>";
	}

	ScopeModifier(Scope scope)
	{
		SetScope(scope);
	}

	Scope GetScope() const { return m_scope; }
	void SetScope(Scope scope)
	{
		if constexpr(Required)
		{
			if (scope == Scope::None)
			{
				std::cerr << "[ERROR] Scope modifier requires non-empty scope" << std::endl;
				std::exit(EXIT_FAILURE);
			}
		}
		m_scope = scope;
	}

	std::string OpCodeModifier() const
	{
		return ScopeString(m_scope);
	}

protected:
	Scope m_scope;
};

}
