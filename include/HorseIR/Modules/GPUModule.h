#pragma once

#include "HorseIR/Tree/Tree.h"

namespace HorseIR {

static Module *GPUModule = new LibraryModule("GPU", {
	// Sort
	new BuiltinFunction(BuiltinFunction::Primitive::GPUOrderInit),
	new BuiltinFunction(BuiltinFunction::Primitive::GPUOrder),

	// Group by
	new BuiltinFunction(BuiltinFunction::Primitive::GPUGroup)
});

}
