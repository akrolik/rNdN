#pragma once

#include <string>

namespace PTX {

#define REQUIRE_SPACE_PARAM(CONTEXT, ENABLED) \
	constexpr static bool SpaceSupported = ENABLED; \
	static_assert(Assert == false || SpaceSupported == true, "PTX::" TO_STRING(CONTEXT) " does not support PTX state space");

#define REQUIRE_SPACE_PARAMS(context, D_ENABLED, T_ENABLED) \
	constexpr static bool SpaceSupported = D_ENABLED && T_ENABLED; \
	static_assert(Assert == false || SpaceSupported == true, "PTX::" TO_STRING(context) " does not support PTX state spaces");

// @struct StateSpace
//
// Storage space used for addressable variables
 
struct StateSpace {
	static std::string Name() { return ".<unknown>"; }

	enum class Kind {
		Register,
		SpecialRegister,
		Addressable,
		Local,
		Global,
		Shared,
		Const,
		Parameter
	};
	virtual Kind GetKind() const = 0;

	// Polymorphism
	virtual ~StateSpace() = default;
};

struct RegisterSpace : StateSpace
{
	static std::string Name() { return ".reg"; }

	StateSpace::Kind GetKind() const override { return StateSpace::Kind::Register; }
}; 

struct SpecialRegisterSpace : RegisterSpace
{
	static std::string Name() { return ".sreg"; }

	StateSpace::Kind GetKind() const override { return StateSpace::Kind::SpecialRegister; }
};	

struct AddressableSpace : StateSpace
{
	StateSpace::Kind GetKind() const override { return StateSpace::Kind::Addressable; }
};

struct LocalSpace : AddressableSpace
{
	static std::string Name() { return ".local"; }

	StateSpace::Kind GetKind() const override { return StateSpace::Kind::Local; }
};

struct GlobalSpace : AddressableSpace
{
	static std::string Name() { return ".global"; }

	StateSpace::Kind GetKind() const override { return StateSpace::Kind::Global; }
};

struct SharedSpace : AddressableSpace
{
	static std::string Name() { return ".shared"; }

	StateSpace::Kind GetKind() const override { return StateSpace::Kind::Shared; }
};

struct ConstSpace : AddressableSpace
{
	static std::string Name() { return ".const"; }

	StateSpace::Kind GetKind() const override { return StateSpace::Kind::Const; }
};
 
struct ParameterSpace : AddressableSpace
{
	static std::string Name() { return ".param"; }

	StateSpace::Kind GetKind() const override { return StateSpace::Kind::Parameter; }
};

}
