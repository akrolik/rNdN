#pragma once

#include "PTX/Operands/Variable.h"
#include "PTX/StateSpaces/MemorySpace.h"
#include "PTX/StateSpaces/ParameterSpace.h"
#include "PTX/StateSpaces/RegisterSpace.h"

namespace PTX {

template<class S, class D>
class SpaceAdapter : public D
{
public:
	SpaceAdapter(S *space) : m_space(space) {}

private:
	S *m_space;
};

template<class S, class D>
using MemorySpaceAdapter = SpaceAdapter<MemorySpace<S>, MemorySpace<D>>;

template<class S, class D>
using ParameterSpaceAdapter = SpaceAdapter<ParameterSpace<S>, ParameterSpace<D>>;

template<class S, class D>
using RegisterSpaceAdapter = SpaceAdapter<RegisterSpace<S>, RegisterSpace<D>>;

}
