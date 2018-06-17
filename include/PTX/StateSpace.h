#pragma once

namespace PTX {

#define REQUIRE_BASE_SPACE(context, space) static_assert(std::is_base_of<space, S>::value, "PTX::" TO_STRING(context) " requires base state space PTX::" TO_STRING(space))
#define REQUIRE_EXACT_SPACE(context, space) static_assert(std::is_same<space, S>::value, "PTX::" TO_STRING(context) " requires exact state space PTX::" TO_STRING(space))

// @struct StateSpace
//
// Storage space used for addressable variables
 
struct StateSpace { static std::string Name() { return ".<unknown>"; } }; 

template<class T>
class Register;

struct RegisterSpace : StateSpace
{
	template<class T>
	using VariableType = Register<T>;

	static std::string Name() { return ".reg"; }
}; 

struct SpecialRegisterSpace : RegisterSpace { static std::string Name() { return ".sreg"; } }; 

template<class T, class S>
class AddressableVariable;

struct AddressableSpace : StateSpace {};

struct LocalSpace : AddressableSpace
{
	template<class T>
	using VariableType = AddressableVariable<T, LocalSpace>;

	static std::string Name() { return ".local"; }
};

struct GlobalSpace : AddressableSpace
{
	template<class T>
	using VariableType = AddressableVariable<T, GlobalSpace>;

	static std::string Name() { return ".global"; }
};

struct SharedSpace : AddressableSpace
{
	template<class T>
	using VariableType = AddressableVariable<T, SharedSpace>;

	static std::string Name() { return ".shared"; }
};

struct ConstSpace : AddressableSpace
{
	template<class T>
	using VariableType = AddressableVariable<T, ConstSpace>;

	static std::string Name() { return ".const"; }
};
 
struct ParameterSpace : AddressableSpace
{
	template<class T>
	using VariableType = AddressableVariable<T, ParameterSpace>;

	static std::string Name() { return ".param"; }
};

}
