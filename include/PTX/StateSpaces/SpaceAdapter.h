#pragma once

#include "PTX/StateSpaces/AddressableSpace.h"
#include "PTX/StateSpaces/RegisterSpace.h"

namespace PTX {

template<class D, class S>
class RegisterSpaceAdapter : public RegisterSpace<D>
{
public:
	RegisterSpaceAdapter(const RegisterSpace<S> *space) : RegisterSpace<D>(space->GetNames()), m_space(space) {}

private:
	const RegisterSpace<S> *m_space;
};

}
