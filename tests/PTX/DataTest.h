#pragma once

#include "Test.h"

#include "PTX/Instructions/Data/ConvertAddressInstruction.h"
#include "PTX/Instructions/Data/ConvertInstruction.h"
#include "PTX/Instructions/Data/IsSpaceInstruction.h"
#include "PTX/Instructions/Data/LoadInstruction.h"
#include "PTX/Instructions/Data/LoadNCInstruction.h"
#include "PTX/Instructions/Data/LoadUniformInstruction.h"
#include "PTX/Instructions/Data/MoveInstruction.h"
#include "PTX/Instructions/Data/PackInstruction.h"
#include "PTX/Instructions/Data/PermuteInstruction.h"
#include "PTX/Instructions/Data/PrefetchInstruction.h"
#include "PTX/Instructions/Data/PrefetchUniformInstruction.h"
#include "PTX/Instructions/Data/ShuffleInstruction.h"
#include "PTX/Instructions/Data/StoreInstruction.h"
#include "PTX/Instructions/Data/UnpackInstruction.h"

namespace Test {

class DataTest : public Test
{
public:
	void Execute()
	{

	}
};

}
