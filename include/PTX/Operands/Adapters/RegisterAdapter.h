#pragma once

#include "PTX/Operands/Variables/Register.h"

namespace PTX {

template<class D, class S>
class RegisterAdapter : public Register<D>
{
public:
	RegisterAdapter(const Register<S> *variable) : Register<D>(variable->GetName()) {}

	json ToJSON() const override
	{
		json j;
		j["kind"] = "PTX::RegisterAdapter";
		j["destination"] = D::Name();
		j["source"] = S::Name();
		j["register"]["kind"] = "PTX::Register";
		j["register"]["name"] = Register<D>::m_name;
		j["register"]["type"] = S::Name();
		j["register"]["space"] = RegisterSpace::Name();
		return j;
	}
};

}
