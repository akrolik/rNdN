#pragma once

#include <string>

namespace PTX {

#define REQUIRE_SPACE_PARAM(CONTEXT, ENABLED) \
	constexpr static bool SpaceSupported = ENABLED; \
	static_assert(Assert == false || SpaceSupported == true, "PTX::" TO_STRING(CONTEXT) " does not support PTX state space");

#define REQUIRE_SPACE_PARAMS(context, D_ENABLED, T_ENABLED) \
	constexpr static bool SpaceSupported = D_ENABLED && T_ENABLED; \
	static_assert(Assert == false || SpaceSupported == true, "PTX::" TO_STRING(context) " does not support PTX state spaces");

template<class T, class S, bool Assert>
class Variable;

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
	template<class T>
	using VariableType = Variable<T, RegisterSpace, true>;

	static std::string Name() { return ".reg"; }
}; 

struct SpecialRegisterSpace : RegisterSpace { static std::string Name() { return ".sreg"; } }; 

struct AddressableSpace : StateSpace
{
	template<class T>
	using VariableType = Variable<T, AddressableSpace, true>;
};

struct LocalSpace : AddressableSpace
{
	template<class T>
	using VariableType = Variable<T, LocalSpace, true>;

	static std::string Name() { return ".local"; }
};

struct GlobalSpace : AddressableSpace
{
	template<class T>
	using VariableType = Variable<T, GlobalSpace, true>;

	static std::string Name() { return ".global"; }
};

struct SharedSpace : AddressableSpace
{
	template<class T>
	using VariableType = Variable<T, SharedSpace, true>;

	static std::string Name() { return ".shared"; }
};

struct ConstSpace : AddressableSpace
{
	template<class T>
	using VariableType = Variable<T, ConstSpace, true>;

	static std::string Name() { return ".const"; }
};
 
struct ParameterSpace : AddressableSpace
{
	template<class T>
	using VariableType = Variable<T, ParameterSpace, true>;

	static std::string Name() { return ".param"; }
};

}
