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

	// Polymorphism
	virtual ~StateSpace() = default;
};

struct RegisterSpace : StateSpace
{
	static std::string Name() { return ".reg"; }
}; 

struct SpecialRegisterSpace : RegisterSpace
{
	static std::string Name() { return ".sreg"; }
};	

struct AddressableSpace : StateSpace {};

struct LocalSpace : AddressableSpace
{
	static std::string Name() { return ".local"; }
};

struct GlobalSpace : AddressableSpace
{
	static std::string Name() { return ".global"; }
};

struct SharedSpace : AddressableSpace
{
	static std::string Name() { return ".shared"; }
};

struct ConstSpace : AddressableSpace
{
	static std::string Name() { return ".const"; }
};
 
struct ParameterSpace : AddressableSpace
{
	static std::string Name() { return ".param"; }
};

}
