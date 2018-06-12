#pragma once

#include "PTX/StateSpace.h"

namespace PTX {

template<class S>
class Resource
{
	REQUIRE_BASE_SPACE(Resource, StateSpace);
};

template<class S>
class ResourceDeclaration
{
	REQUIRE_BASE_SPACE(Resource, StateSpace);
};

}
