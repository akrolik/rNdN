#pragma once

namespace PTX {

enum class LoadSynchronization {
	Weak,
	Volatile,
	Relaxed,
	Acquire
};

enum class StoreSynchronization {
	Weak,
	Volatile,
	Relaxed,
	Release
};

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

}
