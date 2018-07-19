#pragma once

namespace PTX {

#define REQUIRE_SPACE_PARAM(CONTEXT, ENABLED) \
	constexpr static bool SpaceSupported = ENABLED; \
	static_assert(Assert == false || SpaceSupported == true, "PTX::" TO_STRING(CONTEXT) " does not support PTX state space");

// @struct StateSpace
//
// Storage space used for addressable variables
 
struct StateSpace { static std::string Name() { return ".<unknown>"; } }; 

template<class T, class S, typename Enabled>//typename Enabled = void>
class Variable;

struct RegisterSpace : StateSpace
{
	template<class T>
	using VariableType = Variable<T, RegisterSpace, void>;

	static std::string Name() { return ".reg"; }
}; 

struct SpecialRegisterSpace : RegisterSpace { static std::string Name() { return ".sreg"; } }; 

struct AddressableSpace : StateSpace {};

struct LocalSpace : AddressableSpace
{
	template<class T>
	using VariableType = Variable<T, LocalSpace, void>;

	static std::string Name() { return ".local"; }
};

struct GlobalSpace : AddressableSpace
{
	template<class T>
	using VariableType = Variable<T, GlobalSpace, void>;

	static std::string Name() { return ".global"; }
};

struct SharedSpace : AddressableSpace
{
	template<class T>
	using VariableType = Variable<T, SharedSpace, void>;

	static std::string Name() { return ".shared"; }
};

struct ConstSpace : AddressableSpace
{
	template<class T>
	using VariableType = Variable<T, ConstSpace, void>;

	static std::string Name() { return ".const"; }
};
 
struct ParameterSpace : AddressableSpace
{
	template<class T>
	using VariableType = Variable<T, ParameterSpace, void>;

	static std::string Name() { return ".param"; }
};

}
